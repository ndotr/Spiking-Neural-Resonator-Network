import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd

import spinr.metrics.detection
import spinr.metrics.metrics
import spinr.tools.log
import spinr.data.simulation
import spinr.data.pitt_radar
import spinr.net
import spinr.signal_processing.fft
import spinr.signal_processing.preprocess
import spinr.signal_processing.cfar
import spinr.metrics.snr

log = spinr.tools.log.terminal_log()


radar_data = spinr.data.pitt_radar.load_data("/home/nreeb/data/pitt-radar/numpy", frame_idx=5000)
radar_data = radar_data[:,:,:,:1]
n_frames, n_chirps, n_samples, n_rx = radar_data.shape

# Network
params = [
            0.0,        # alpha_x smooth, default: 0.6
            1.0,        # alpha_l, default: 1
            0.1,        # alpha_a_smooth, default: 0.1
            0.0001,     # alpha_grad_smooth, default: 0.0001
            16,         # t_start
            0.001,       # alpha_u_rate, default: 0.01
            1000,        # u_rate_threshold
            -100           # u_rate_rest
]

net = spinr.net.SpiNRNet(n_antennas=n_rx, n_distances=n_samples, n_angles=n_rx, n_velocities=n_chirps, 
                         params=params, log=log, kernel='rate')
gpu = net.forward_gpu(radar_data, debug=True)

fft_data = spinr.signal_processing.fft.range_doppler_fft(radar_data=radar_data)

fig, axs = plt.subplots(ncols=2)
axs[0].imshow(gpu[0,:,:,0,-1,0].T, aspect='auto')
axs[1].imshow(fft_data[0,:,:,0].T, aspect='auto')
plt.savefig('figures/tests/test_rate_kernel.pdf')


fig, axs = plt.subplots(nrows=3)
axs[0].plot(radar_data[0,-1,:,0].T)
axs[1].plot(net.states[0,100:110,48,0,-1,:,0].T)
axs[2].plot(net.states[0,100:110,48,0,-1,:,1].T)
plt.savefig('figures/tests/test_rate_kernel_states.pdf')
