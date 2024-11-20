import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd

import spinr.metrics.detection
import spinr.metrics.metrics
import spinr.tools.log
import spinr.data.simulation
import spinr.data.infineon_bbm
import spinr.net
import spinr.signal_processing.fft
import spinr.signal_processing.preprocess
import spinr.signal_processing.cfar
import spinr.metrics.snr
import config_net as config
import config_fft

log = spinr.tools.log.terminal_log()

dataset_path = "data/level0_random76x16x8_1dmax_0010rcs_2targets_23deltad_23deltaa_32chirps/seed_1/"

radar_config = spinr.data.infineon_bbm.load_config(dataset_path)

radar_data = spinr.data.infineon_bbm.load_data(dataset_path + "data_020.mat")
targets = spinr.data.infineon_bbm.load_targets(dataset_path + "targets_020.mat")[0]

n_frames, n_chirps, n_samples, n_rx = radar_data.shape

# Network
net = spinr.net.SpiNRNet(n_antennas=n_rx, n_distances=n_samples//2, n_angles=n_rx, n_velocities=1, 
                         params=config.params, log=log)
gpu = net.forward_gpu(radar_data, debug=False)

# pre processing
preprocessed_data = spinr.signal_processing.preprocess.subtract_exp_filter(radar_data, alpha=config_fft.exp_alpha)
# FFT
fft_data = spinr.signal_processing.fft.range_angle_fft(preprocessed_data, remove_zero_modes=config_fft.remove_zero_modes)
# post processing
idx = fft_data < config_fft.threshold
fft_data[idx] = 0
fft_data = fft_data[:,-1]
fft_data = cp.array(fft_data)

# Targets
target_maps = spinr.data.infineon_bbm.targets_to_maps(targets, n_distances=n_samples//2, n_angles=n_rx, config=radar_config, rcs_flag=False)
target_maps = cp.array(target_maps)

# Extract data for CFAR
data = gpu[...,0] # extract only grad data
data[:,:,:config.remove_zero_modes,:,:] = 0 # remove zero modes
data = cp.array(data[:,-1,...,-1]) # Get fixed velocity (last) and fixed chirp (last)

# SSR Metrics
snr = spinr.metrics.snr.snr_gpu(data, target_maps, kernel=None, tags=0)
fft_snr = spinr.metrics.snr.snr_gpu(fft_data, target_maps, kernel=None, tags=0)

# CFAR Metrics
kernel = cp.ones([3,5])
kernel[3//2,5//2] = 0
detection_area = cp.ones((3,1)) 

factors = cp.round(cp.arange(2.2, 2.6, 0.1),2) #  
offsets = cp.round(cp.arange(0.075, 0.080, 0.005),4) #
ca_maps = spinr.signal_processing.cfar.create_ca_maps_gpu(data, kernel)
cfar_entries = spinr.metrics.detection.cacfar_detection_gpu(data, target_maps, ca_maps, factors, offsets, detection_area=detection_area)

factors = cp.round(cp.arange(1.2, 1.8, 0.1),2) #  
offsets = cp.round(cp.arange(0.055, 0.080, 0.005),4) #
fft_ca_maps = spinr.signal_processing.cfar.create_ca_maps_gpu(fft_data, kernel)
fft_cfar_entries = spinr.metrics.detection.cacfar_detection_gpu(fft_data, target_maps, fft_ca_maps, factors, offsets, detection_area=detection_area)

df = pd.DataFrame(np.vstack(cfar_entries), columns=["batch", "target", "offset", "factor", "precision", "recall", "false positive", "false negative"])
df.to_csv('results/tests/cfar_eval.csv', sep='|', index=False, encoding='utf-8')
df = pd.DataFrame(np.vstack(snr), columns=["batch", "target", "ssr1", "ssr2"])
df.to_csv('results/tests/snr_eval.csv', sep='|', index=False, encoding='utf-8')

df = pd.DataFrame(np.vstack(fft_cfar_entries), columns=["batch", "target", "offset", "factor", "precision", "recall", "false positive", "false negative"])
df.to_csv('results/tests/fft_cfar_eval.csv', sep='|', index=False, encoding='utf-8')
df = pd.DataFrame(np.vstack(fft_snr), columns=["batch", "target", "ssr1", "ssr2"])
df.to_csv('results/tests/fft_snr_eval.csv', sep='|', index=False, encoding='utf-8')