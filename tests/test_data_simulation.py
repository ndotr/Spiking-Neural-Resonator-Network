import numpy as np
import matplotlib.pyplot as plt
import spinr.data.simulation

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

plt.figure()
plt.plot(np.real(data).T)

plt.savefig("figures/tests/test_data_simulation.pdf")
plt.savefig("figures/tests/test_data_simulation.png")