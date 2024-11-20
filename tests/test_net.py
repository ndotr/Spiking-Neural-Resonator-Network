import numpy as np
import matplotlib.pyplot as plt
import spinr.tools.log
import spinr.data.simulation
import spinr.net
import config_net as config
log = spinr.tools.log.terminal_log()

# Example parameters for simulation
n_sensors = 8  # Number of sensors in the array
n_samples = 512  # Number of time steps (samples)
sampling_rate = 10000  # Sampling rate (Hz)
noise_power = 0.1  # Noise power density

n_sources = 4 # Number of signal sources
source_freqs = [1000]*n_sources # Signal frequency (Hz)
doa_deg = np.random.uniform(-80, 80, n_sources)
doa_rad = np.radians(doa_deg)
source_amps = np.random.uniform(1,8,(n_sources,1))

# Data generation
source_signals = spinr.data.simulation.create_source_signal(n_sources=n_sources, source_freqs=source_freqs, 
                                                            n_samples=n_samples, sampling_rate=sampling_rate, 
                                                            source_amps=source_amps)
# Create a uniform linear array of sensors
# distance between sensors = wavelentgh /2
rx_signals = spinr.data.simulation.create_rx_signal(source_signals=source_signals, n_sensors=n_sensors, doa_rad=doa_rad)
data = spinr.data.simulation.add_white_noise(rx_signals, noise_power=noise_power)
data = np.swapaxes(np.real(data), 1, 0)
data = data.reshape(1,1,512,8)
data = data.repeat(2, axis=0)


# Network
net = spinr.net.SpiNRNet(n_antennas=n_sensors, n_distances=256, n_angles=n_sensors, n_velocities=1, 
                         params=config.params, log=log)

cpu = net.forward_cpu(data, debug=False)
net.reset()
gpu = net.forward_gpu(data, debug=False)

fig, axs = plt.subplots(nrows=3, ncols=5, sharex=True, figsize=(16,9))
for i in range(5):
    axs[0,i].set_title("Spiking Function {}\nCPU".format(i))
    axs[0,i].plot(cpu[-1,0,:,i,0,i].T)
    axs[1,i].set_title("GPU")
    axs[1, i].plot(gpu[-1,0,:,i,0,i].T)
    axs[2,i].set_title("Error")
    axs[2, i].plot(np.abs(gpu[-1,0,:,i,0,i].T - cpu[-1,0,:,i,0,i].T))
plt.tight_layout()
plt.savefig('figures/tests/test_net.png')