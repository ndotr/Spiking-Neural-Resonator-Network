import numpy as np
import h5py
import tensorflow as tf

import tempfile

try:
    import torch
    from torch.utils.data import Dataset
    __has_torch = True
except ImportError:
    __has_torch = False


import mmwave

import logging

logger = logging.getLogger()

def read_radar_params(filename):
    """Reads a text file containing serial commands and returns parsed config as a dictionary"""
    with open(filename) as cfg:
        iwr_cmds = cfg.readlines()
        iwr_cmds = [x.strip() for x in iwr_cmds]
        radar_cfg = parse_commands(iwr_cmds)

    logger.debug(radar_cfg)
    return radar_cfg

def parse_commands(commands):
    """Calls the corresponding parser for each command in commands list"""
    cfg = None
    for line in commands:
        try:
            cmd = line.split()[0]
            args = line.split()[1:]
            cfg = command_handlers[cmd](args, cfg)
        except KeyError:
            logger.debug(f'{cmd} is not handled')
        except IndexError:
            logger.debug(f'line is empty "{line}"')
    return cfg

def dict_to_list(cfg):
    """Generates commands from config dictionary"""
    cfg_list = ['flushCfg','dfeDataOutputMode 1']

    # rx antennas/lanes for channel config
    rx_bool = [cfg['rx4'], cfg['rx3'], cfg['rx2'], cfg['rx1']]
    rx_mask = sum(2 ** i for i, v in enumerate(reversed(rx_bool)) if v)
    # number of tx antennas for channel config
    tx_bool = [cfg['tx3'], cfg['tx2'], cfg['tx1']]
    tx_mask = sum(2 ** i for i, v in enumerate(reversed(tx_bool)) if v)
    #print('[NOTE] Azimuth angle can be determined from channel config.') if cfg['tx2'] is True and (cfg['tx1'] or cfg['tx3']) is False else 0
    #print('[NOTE] Azimuth angle can be determined from channel config.') if cfg['tx2'] is False and (cfg['tx1'] or cfg['tx3']) is True else 0
    #print('[NOTE] Elevation and Azimuth angle can be determined from channel config.') if cfg['tx2'] is True and (cfg['tx1'] or cfg['tx3']) else 0
    cfg_list.append('channelCfg %s %s 0' % (rx_mask, tx_mask))  # rx and tx mask

    # adc config
    if cfg['isComplex'] and cfg['image_band']:
        outputFmt = 2
        #print('[NOTE] Complex 2x mode, both Imaginary and Real IF spectrum is filtered and sent to ADC, so\n'
        #      '       if Sampling rate is X, ADC data would include frequency spectrum from -X/2 to X/2.')
    elif cfg['isComplex'] and not cfg['image_band'] == True:
        outputFmt = 1
        #print('[NOTE] Complex 1x mode, Only Real IF Spectrum is filtered and sent to ADC, so if Sampling rate\n'
        #      '       is X, ADC data would include frequency spectrum from 0 to X.')
    else: raise ValueError("Real Data Type Not Supported")
    cfg_list.append('adcCfg 2 %s' % outputFmt)  # 16 bits (mandatory), complex 1x or 2x

    # adc power
    if cfg['adcPower'] =='low':
        power_mode = 1
        #print('[NOTE] The Low power ADC mode limits the sampling rate to half the max value.')
    elif cfg['adcPower'] =='regular': power_mode = 0
    else: raise ValueError("ADC power level Not Supported")
    cfg_list.append('lowPower 0 %s' % power_mode)  # power mode

    # profile configs
    for profile_ii in cfg['profiles']:
        cfg_list.append('profileCfg %s %s %s %s %s %s %s %s %s %s %s %s %s %s'
                % (profile_ii['id'],
                float(profile_ii['start_frequency']/1e9),
                float(profile_ii['idle']/1e-6),
                float(profile_ii['adcStartTime']/1e-6),
                float(profile_ii['rampEndTime']/1e-6),
                int(profile_ii['txPower']),
                int(profile_ii['txPhaseShift']),
                float(profile_ii['freqSlopeConst']/1e12),
                float(profile_ii['txStartTime']/1e-6),
                int(profile_ii['adcSamples']),
                int(profile_ii['adcSampleRate']/1e3),
                int(profile_ii['hpfCornerFreq1']),
                int(profile_ii['hpfCornerFreq2']),
                int(profile_ii['rxGain'])))

    # chirp configs
    for chirp_ii in cfg['chirps']:

        # Check if chirp is referring to valid profile config
        profile_valid = False
        for profile_ii in cfg['profiles']:
            if chirp_ii['profileID'] == profile_ii['id']: profile_valid = True
        if profile_valid is False: raise ValueError("The following profile id used in chirp "
                                                    "is invalid: %i" % chirp_ii['profileID'])
        ###############################################################################################################
        '''
        # check if tx values are valid
        if hamming([chirp_ii['chirptx3'],chirp_ii['chirptx2'],chirp_ii['chirptx1']],
            [cfg['tx3'], cfg['tx2'], cfg['tx1']])*3 > 1:
            raise ValueError("Chirp should have at most one different Tx than channel cfg")
        '''
        ###############################################################################################################
        if chirp_ii['chirpStartIndex'] > chirp_ii['chirpStopIndex']: raise ValueError("Particular chirp start index after chirp stop index")
        tx_bool = [chirp_ii['chirptx3'],chirp_ii['chirptx2'],chirp_ii['chirptx1']]
        tx_mask = sum(2 ** i for i, v in enumerate(reversed(tx_bool)) if v)
        cfg_list.append('chirpCfg %s %s %s %s %s %s %s %s'
                % (chirp_ii['chirpStartIndex'],
                   chirp_ii['chirpStopIndex'],
                   chirp_ii['profileID'],
                   chirp_ii['startFreqVariation'],
                   chirp_ii['slopeVariation'],
                   chirp_ii['idleVariation'],
                   chirp_ii['adcStartVariation'],
                   tx_mask))

    # frame config
    chirpStop = 0
    chirpStart = 511  # max value for chirp start index
    for chirp_ii in cfg['chirps']:
        chirpStop = max(chirpStop, chirp_ii['chirpStopIndex'])
        chirpStart = min(chirpStart,chirp_ii['chirpStartIndex'])
    chirps_len  = chirpStop + 1

    numLoops = cfg['numChirps']/chirps_len
    if chirpStart > chirpStop: raise ValueError("Chirp(s) start index is after chirp stop index")
    if numLoops % 1 != 0: raise ValueError("Number of loops is not integer")
    if numLoops > 255 or numLoops < 1: raise ValueError("Number of loops must be int in [1,255]")

    numFrames = cfg['numFrames'] if 'numFrames' in cfg.keys() else 0  # if zero => inf

    cfg_list.append('frameCfg %s %s %s %s %s 1 0'
            % (chirpStart, chirpStop, int(numLoops), numFrames, 1000/cfg['fps']))

    cfg_list.append('testFmkCfg 0 0 0 1')
    cfg_list.append('setProfileCfg disable ADC disable')
    return cfg_list


def channelStr_to_dict(args, curr_cfg=None):
    """Handler for `channelcfg`"""

    if curr_cfg:
        cfg = curr_cfg
    else:
        cfg = {}

    # This is the number of receivers which is equivalent to the number of lanes in the source code
    # Later, may include the result from the number of transmitters
    rx_bin = bin(int(args[0]))[2:].zfill(4)
    cfg['numLanes'] = len([ones for ones in rx_bin if ones == '1'])
    (cfg['rx4'],cfg['rx3'],cfg['rx2'],cfg['rx1']) = [bool(int(ones)) for ones in rx_bin]

    # This is the number of transmitters
    tx_bin = bin(int(args[1]))[2:].zfill(3)
    cfg['numTx'] = len([ones for ones in tx_bin if ones == '1'])
    (cfg['tx3'], cfg['tx2'], cfg['tx1']) = [bool(int(ones)) for ones in tx_bin]
    #print('[NOTE] Azimuth angle can be determined from channel config.') if cfg['tx2'] is True and (cfg['tx1'] or cfg['tx3']) is False else 0
    #print('[NOTE] Azimuth angle can be determined from channel config.') if cfg['tx2'] is False and (cfg['tx1'] or cfg['tx3']) is True else 0
    #print('[NOTE] Elevation and Azimuth angle can be determined from channel config.') if cfg['tx2'] is True and (cfg['tx1'] or cfg['tx3']) else 0


    return cfg


def profileStr_to_dict(args, curr_cfg=None):
    """Handler for `profileCfg`"""
    normalizer = [None, 1e9, 1e-6, 1e-6, 1e-6, None, None, 1e12, 1e-6, None, 1e3, None, None, None]
    dtype = [int, float, float, float, float, float, float, float, float, int, float, int, int, float]
    keys = ['id',
            'start_frequency',
            'idle',
            'adcStartTime',
            'rampEndTime',
            'txPower',
            'txPhaseShift',
            'freqSlopeConst',
            'txStartTime',
            'adcSamples',
            'adcSampleRate',
            'hpfCornerFreq1',
            'hpfCornerFreq2',
            'rxGain',
            ]
    # Check if the main dictionary exists
    if curr_cfg:
        cfg = curr_cfg
        if 'profiles' not in cfg.keys():
            cfg['profiles']=[]
    else:
        cfg = {'profiles': []}

    profile_dict = {}
    for k, v, n, d in zip(keys, args, normalizer, dtype):
        profile_dict[k] = d(float(v) * n if n else v)

    cfg['profiles'].append(profile_dict)
    return cfg


def chirp_to_dict(args,curr_cfg=None):
    """Handler for `chirpCfg`"""
    if curr_cfg:
        cfg = curr_cfg
        if 'chirps' not in cfg.keys():
            cfg['chirps'] = []
    else:
        cfg = {'chirps': []}

    chirp_dict = {}
    chirp_dict['chirpStartIndex'] = int(args[0])
    chirp_dict['chirpStopIndex'] = int(args[1])
    chirp_dict['profileID'] = int(args[2])
    chirp_dict['startFreqVariation'] = float(args[3])
    chirp_dict['slopeVariation'] = float(args[4])
    chirp_dict['idleVariation'] = float(args[5])
    chirp_dict['adcStartVariation'] = float(args[6])

    tx_bin = bin(int(args[7]))[2:].zfill(3)
    (chirp_dict['chirptx3'], chirp_dict['chirptx2'], chirp_dict['chirptx1']) = [bool(int(ones)) for ones in tx_bin]

    cfg['chirps'].append(chirp_dict)
    return cfg


def power_to_dict(args,curr_cfg=None):
    """handler for `lowPower`"""
    if curr_cfg:
        cfg = curr_cfg
    else:
        cfg = {}
    if int(args[1]) ==1:
        cfg['adcPower'] = 'low'
        #print('[NOTE] The Low power ADC mode limits the sampling rate to half the max value.')
    elif int(args[1]) ==0:
        cfg['adcPower'] = 'regular'
    else:
        raise ValueError ("Invalid Power Level")
    return cfg


def frameStr_to_dict(args, cfg):
    """Handler for `frameCfg`"""

    # Number of chirps
    if 'chirps' not in cfg.keys():
        raise ValueError("Need to define chirps before frame")

    chirpStop =0
    for ii in range(len(cfg['chirps'])):
        chirpStop = max(chirpStop,cfg['chirps'][ii]['chirpStopIndex'])
    chirps_len = chirpStop + 1

    cfg['numChirps'] = int(args[2]) * chirps_len  # num loops * len(chirps)
    if int(args[3]) != 0: cfg['numFrames'] = int(args[3])

    # args[4] is the time in milliseconds of each frame
    cfg['fps'] = 1000/float(args[4])


    return cfg


def adcStr_to_dict(args, curr_cfg=None):
    """Handler for `adcCfg`"""
    if curr_cfg:
        cfg = curr_cfg
    else:
        cfg = {}

    if int(args[1]) == 1:
        cfg['isComplex'] = True
        cfg['image_band'] = False
        #print('[NOTE] Complex 1x mode, Only Real IF Spectrum is filtered and sent to ADC, so if Sampling rate\n'
        #      '       is X, ADC data would include frequency spectrum from 0 to X.')
    elif int(args[1]) == 2:
        cfg['isComplex'] = True
        cfg['image_band'] = True
        #print('[NOTE] Complex 2x mode, both Imaginary and Real IF spectrum is filtered and sent to ADC, so\n'
        #      '       if Sampling rate is X, ADC data would include frequency spectrum from -X/2 to X/2.')
    else:
        raise ValueError("Real Data Type Not Supported")

    return cfg

#Mapping of serial command to command handler
command_handlers = {
    'channelCfg': channelStr_to_dict,
    'profileCfg': profileStr_to_dict,
    'chirpCfg': chirp_to_dict,
    'frameCfg': frameStr_to_dict,
    'adcCfg': adcStr_to_dict,
    'lowPower': power_to_dict,
}

class H5DatasetIterator():
    """Iterates through aligned frames dataset"""
    def __init__(self, dset, streams):
        super().__init__()
        self._dset = dset
        self._idx = 0
        self.req_streams = streams

    def __iter__(self):
        return self

    def __next__(self):
        self._idx += 1
        try:
            return tuple(self._dset[s][self._idx] for s in self.req_streams)
        except IndexError:
            self._idx = 0
            raise StopIteration

    def __len__(self):
        return len(self._dset)

class H5DatasetLoader(object):
    """A thin wrapper around h5py to provide convenience functions for training"""
    def __init__(self, filenames, default_streams=None):
        super(H5DatasetLoader, self).__init__()
        self.filenames = filenames
        if isinstance(self.filenames, list):
            self._h5_tempfile = tempfile.NamedTemporaryFile()
            #self.h5_file = h5py.File(self._h5_tempfile, 'w', libver='latest')

            self._allfiles, _allstreams, _lengths = zip(*[H5DatasetLoader.load_single_h5(f) for f in self.filenames])

            total_len = sum(_lengths)

            #create virtual datasets of, assumes that all files have the streams of first file and shape of first file
            ll = (0,) + _lengths
            ll = np.cumsum(ll)
            for s in _allstreams[0]:
                shape = (total_len, ) + self._allfiles[0][s].shape[1:]
                layout = h5py.VirtualLayout(shape=shape, dtype=self._allfiles[0][s].dtype)

                for idx, f in enumerate(self._allfiles):
                    vsource = h5py.VirtualSource(f[s])
                    layout[ll[idx]:ll[idx+1]] = vsource

                with h5py.File(self._h5_tempfile.name, 'a', libver='latest') as f:
                    f.create_virtual_dataset(s, layout,)
            self._h5_tempfile.flush()
            self.h5_file = H5DatasetLoader.load_single_h5(self._h5_tempfile.name)[0]
        else:
            self.h5_file = H5DatasetLoader.load_single_h5(self.filenames)[0]
        self.streams_available = list(self.h5_file.keys())
        self.default_streams = default_streams

        if default_streams is not None:
            for s in default_streams:
                assert s in self.streams_available, f"{s} not found in available streams"

    @staticmethod
    def load_single_h5(filename):
        h5 = h5py.File(filename, 'r')
        streams_available = list(h5.keys())
        dataset_len = len(h5[streams_available[0]])
        return h5, streams_available, dataset_len

    def __len__(self):
        return len(self.h5_file[self.streams_available[0]])

    def __getitem__(self, stream):
        return self.h5_file[stream]

    def get_iterator(self, streams=None):
        """The default iterator includes all available streams in the order available on the h5 file"""
        if not streams:
            streams = self.default_streams.copy() if self.default_streams is not None else self.streams_available.copy()
        return H5DatasetIterator(self, streams)

    def __iter__(self):
        return self.get_iterator()

    @property
    def filename(self):
        return self.filenames

    def get_torch_dataset(self, streams=None):
        try:
            return RadicalDatasetTorch(self, streams)
        except NameError:
            raise RuntimeError('Torch is not available')


    def get_tf_dataset(self,
                       streams=None,
                       shuffle=False,
                       repeat=False,
                       batchsize=16,
                       preprocess_chain=None,
                       prefetch=2,
                       flatten_single=False,
                      ):
        logger.debug("Tensorflow Dataset creation")
        if streams is None:
            streams = ['radar', 'rgb', 'depth']

        out_shapes = tuple([
            tf.TensorShape(list(self.h5_file[s].shape[1:])) for s in streams
        ])
        out_types = tuple([self.h5_file[s].dtype for s in streams])

        def _gen():
            for i in range(len(self.h5_file[streams[0]])):
                yield tuple(self.h5_file[s][i] for s in streams)

        _dataset = tf.data.Dataset.from_generator(
            _gen,
            output_types = out_types,
            output_shapes = out_shapes,
        )

        if shuffle:
            logger.debug("  Outputs of dataset will be shuffled")
            _dataset = _dataset.shuffle(batchsize * 4)

        if repeat:
            logger.debug(f'  Dataset will be repeated {repeat} files')
            _dataset = _dataset.repeat(repeat)

        if preprocess_chain is not None:
            for op in preprocess_chain:
                _dataset = _dataset.map(op)

        if flatten_single:
            assert(len(streams) == 1)
            logger.debug("  Flattening shapes for single stream inference")
            logger.debug(_dataset)
            _dataset = _dataset.map(lambda x: x)
            logger.debug(_dataset)

        _dataset = _dataset.batch(batchsize)


        if prefetch:
            _dataset = _dataset.prefetch(prefetch)

        return _dataset

# Cell
if __has_torch:
    class RadicalDatasetTorch(Dataset):
        def __init__(self, src_dataset, streams=None, transforms=None):
            self.__src_dataset = src_dataset
            if streams is not None:
                self.__streams = streams
            else:
                self.__streams = ['radar', 'rgb', 'depth']

        def __len__(self):
            return len(self.__src_dataset)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            sample = {s:self.__src_dataset[s][idx, ...] for s in self.__streams}
            return sample
