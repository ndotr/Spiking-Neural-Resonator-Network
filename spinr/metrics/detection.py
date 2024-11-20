import numpy as np
import cupy as cp
import cupyx.scipy.signal
import spinr.signal_processing.cfar

def cacfar_detection_gpu(data, target_maps, ca_maps, factors, offsets, detection_area=None, normalize=True, tags=0):


    entries = np.zeros((len(factors)*len(offsets)*target_maps.shape[0], 8))

    target_maps = target_maps[...,:200,:]
    data = data[...,:200,:]
    ca_maps = ca_maps[...,:200,:]

    if normalize:
        data = cp.einsum('ijk,i->ijk', data, 1/np.max(data,axis=(1,2)))

    for n, factor in enumerate(factors):
        skip = False
        for m, offset in enumerate(offsets):


            if not skip:
                cfar_maps = (ca_maps+offset)*factor
                peak_maps = cp.array(data > (cfar_maps), dtype='float')

                if detection_area is not None:
                    extended_peak_maps = cp.zeros_like(peak_maps)
                    for i in range(peak_maps.shape[0]):
                        extended_peak_maps[i] = cupyx.scipy.signal.convolve2d(peak_maps[i], detection_area, mode='same')>0
                    extended_peak_maps = cp.repeat(cp.expand_dims(extended_peak_maps, axis=0), target_maps.shape[0], axis=0)

                    peak_maps = cp.repeat(cp.expand_dims(peak_maps, axis=0), target_maps.shape[0], axis=0)
                    tp_maps = (extended_peak_maps * target_maps) > 0 # peak is there and target is there
                else:
                    # Detection rate and false detection rate
                    peak_maps = cp.repeat(cp.expand_dims(peak_maps, axis=0), target_maps.shape[0], axis=0)
                    tp_maps = (peak_maps * target_maps) > 0 # peak is there and target is there

                fp_maps = (peak_maps - target_maps) > 0 # peak is there, but no target
                fn_maps = (target_maps - peak_maps) > 0 # target is there, but no peak


                recall = cp.sum(cp.sum(tp_maps,axis=(2,3))/(cp.sum(target_maps, axis=(2,3))+1e-8), axis=1)/target_maps.shape[1]

                fp = cp.sum(cp.sum(fp_maps, axis=(2,3))/cp.clip(cp.sum(peak_maps,axis=(2,3)), a_min=1, a_max=None), axis=1)/target_maps.shape[1]

                fn = cp.sum(cp.sum(fn_maps, axis=(2,3))/cp.clip(cp.sum(target_maps,axis=(2,3)), a_min=1, a_max=None), axis=1)/target_maps.shape[1]

                precision = cp.sum(cp.sum(tp_maps,axis=(2,3))/(cp.sum(peak_maps, axis=(2,3))+1e-8), axis=1)/target_maps.shape[1]

                for target_idx in range(target_maps.shape[0]):
                    entries[m*len(factors)*target_maps.shape[0]+n*target_maps.shape[0]+target_idx] = \
                                                    np.array([tags, target_idx, offset.get(), factor.get(), 
                                                    precision[target_idx].get(), recall[target_idx].get(), 
                                                    fp[target_idx].get(), fn[target_idx].get()])
                if np.sum(peak_maps) < 1:
                    skip = True
            else:
                for target_idx in range(target_maps.shape[0]):
                    entries[m*len(factors)*target_maps.shape[0]+n*target_maps.shape[0]+target_idx] = \
                                                    np.array([tags, target_idx, offset.get(), factor.get(), 
                                                    0, 0, 
                                                    0, 1])

    return entries