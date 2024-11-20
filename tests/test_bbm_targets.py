import numpy as np
import matplotlib.pyplot as plt

import spinr.tools.log
import spinr.data.simulation
import spinr.data.infineon_bbm
import spinr.net
import spinr.signal_processing.fft
import spinr.signal_processing.preprocess
import config_net as config
import config_fft

log = spinr.tools.log.terminal_log()

dataset_path = "data/level0_random76x16x8_1dmax_0010rcs_2targets_23deltad_23deltaa_32chirps/seed_1/"

radar_config = spinr.data.infineon_bbm.load_config(dataset_path)

radar_data = spinr.data.infineon_bbm.load_data(dataset_path + "data_050.mat")
targets = spinr.data.infineon_bbm.load_targets(dataset_path + "targets_050.mat")[0]

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

# Targets
target_maps = spinr.data.infineon_bbm.targets_to_maps(targets, n_distances=n_samples//2, n_angles=n_rx, config=radar_config, rcs_flag=False)

fig, axs = plt.subplots(nrows=3, ncols=8, figsize=(16,9))
for i in range(8):
    axs[0,i].set_title('Targets')
    axs[0,i].imshow(target_maps[0,48+i, 160:180], cmap='Greys', aspect='auto')
    axs[1,i].set_title('SpiNR')
    axs[1,i].imshow(gpu[48+i,0,160:180,:,0,0], cmap='Greys', aspect='auto')
    axs[2,i].set_title('FFT')
    axs[2,i].imshow(fft_data[48+i,0,160:180,:], cmap='Greys', aspect='auto')
plt.tight_layout()
plt.savefig('figures/tests/test_bbm_targets.pdf')