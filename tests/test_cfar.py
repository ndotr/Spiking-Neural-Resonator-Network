import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

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
fft_data = fft_data[:,-1] # only last chirp
fft_data = cp.array(fft_data)

# Targets
target_maps = spinr.data.infineon_bbm.targets_to_maps(targets, n_distances=n_samples//2, n_angles=n_rx, config=radar_config, rcs_flag=False)
target_maps = cp.array(target_maps)

# Extract data for CFAR
data = gpu[...,0] # extract only grad data
data[:,:,:config.remove_zero_modes,:,:] = 0 # remove zero modes
data = cp.array(data[:,-1,...,-1]) # Get fixed velocity (last) and fixed chirp (last)

# SSR
#ssr = spinr.metrics.snr.snr_pertarget_gpu(data, target_maps, kernel=None, tags=0)

# CFAR
kernel = cp.ones([3,5])
kernel[3//2,5//2] = 0
detection_area = cp.ones((3,1)) 

factor = 2.2
offset = 0.075
ca_maps = spinr.signal_processing.cfar.create_ca_maps_gpu(data, kernel)
peak_maps = spinr.signal_processing.cfar.cacfar_detection_gpu(data, ca_maps, factor, offset, detection_area=detection_area)

factor = 1.8
offset = 0.07
fft_ca_maps = spinr.signal_processing.cfar.create_ca_maps_gpu(fft_data, kernel)
fft_peak_maps = spinr.signal_processing.cfar.cacfar_detection_gpu(fft_data, fft_ca_maps, factor, offset, detection_area=detection_area)


fig, axs = plt.subplots(nrows=5, ncols=8, figsize=(16,9))
for i in range(8):
    axs[0,i].set_title('Targets')
    axs[0,i].imshow(target_maps[0,32+8*i, :].get(), cmap='Greys', aspect='auto')
    axs[1,i].set_title('SpiNR')
    axs[1,i].imshow(data[32+8*i,:,:].get(), cmap='Greys', aspect='auto')
    axs[2,i].set_title('Peak Maps')
    axs[2,i].imshow(peak_maps[32+8*i,:,:].get(), cmap='Greys', aspect='auto')
    axs[3,i].set_title('FFT')
    axs[3,i].imshow(fft_data[32+8*i,:,:].get(), cmap='Greys', aspect='auto')
    axs[4,i].set_title('FFT Peak Maps')
    axs[4,i].imshow(fft_peak_maps[32+8*i,:,:].get(), cmap='Greys', aspect='auto')
plt.tight_layout()
plt.savefig('figures/tests/test_cfar.png')