import numpy as np
import matplotlib.pyplot as plt
import spinr.net
import spinr.data.simulation
import spinr.signal_processing.fft
import spinr.signal_processing.preprocess
import spinr.tools.log
import config_fft as config
import config_net
log = spinr.tools.log.terminal_log()

# Example parameters for simulation
n_sensors = 8  # Number of sensors in the array
n_samples = 512  # Number of time steps (samples)
#signal_freq = 0  # Signal frequency (Hz)
sampling_rate = 10000  # Sampling rate (Hz)
noise_power = 0.1  # Noise power density

n_sources = 4 # Number of signal sources
source_freqs = [1000]*n_sources # Signal frequency (Hz)
doa_deg = np.random.uniform(-80, 80, n_sources)
doa_rad = np.radians(doa_deg)
source_amps = np.random.uniform(1,8,(n_sources,1))

# Data generation
source_signals = spinr.data.simulation.create_source_signal(n_sources=n_sources, source_freqs=source_freqs, n_samples=n_samples, sampling_rate=sampling_rate, source_amps=source_amps)
# Create a uniform linear array of sensors
# distance between sensors = wavelentgh /2
rx_signals = spinr.data.simulation.create_rx_signal(source_signals=source_signals, n_sensors=n_sensors, doa_rad=doa_rad)
data = spinr.data.simulation.add_white_noise(rx_signals, noise_power=noise_power)

data = np.swapaxes(np.real(data), 1, 0)
data = data.reshape(1,1,512,8)
data = data.repeat(2, axis=0)

# pre processing
preprocessed_data = spinr.signal_processing.preprocess.subtract_exp_filter(data, alpha=config.exp_alpha)
# FFT
fft_data = spinr.signal_processing.fft.range_angle_fft(preprocessed_data, remove_zero_modes=config.remove_zero_modes)
# post processing
idx = fft_data < config.threshold
fft_data[idx] = 0

# Network
net = spinr.net.SpiNRNet(n_antennas=n_sensors, n_distances=256, n_angles=n_sensors, n_velocities=1, 
                         params=config_net.params, log=log)
gpu = net.forward_gpu(data, debug=False)


fig, axs = plt.subplots(ncols=2)
axs[0].set_title('FFT')
axs[0].imshow(fft_data[0,0,:,:], aspect='auto')
axs[1].set_title('SpiNR')
axs[1].imshow(gpu[0,0,:,:,0,0], aspect='auto')
plt.tight_layout()
plt.savefig('figures/tests/test_fft.png')
