import numpy as np
np.random.seed(0)

def create_source_signal(n_sources, source_freqs, n_samples, sampling_rate, source_amps):
    '''
    Creating IF source signals.
    '''

    # Time vector
    t = np.arange(n_samples) / sampling_rate

    # Generate source signals (sine waves with different frequencies and DOAs)
    source_signals = np.zeros((n_sources, n_samples), dtype=complex)  # Create empty signal matrix
    for i in range(n_sources):
        source_signals[i, :] = source_amps[i]*np.exp(1j * 2 * np.pi * source_freqs[i] * t)  # Simulate signals as complex exponentials

    return source_signals

def create_rx_signal(source_signals, n_sensors, doa_rad):
    '''
    Creating RX signals.
    '''

    n_sources, n_samples = source_signals.shape

    # Combine the signals at the array with phase shifts due to the DOAs
    rx_signals = np.zeros((n_sensors, n_samples), dtype=complex)  # Create array signal matrix
    for i in range(n_sensors):
        for j in range(n_sources):
            # Apply phase shift based on the angle of arrival and sensor position
            # phase_shift = np.exp(-1j * 2 * np.pi * array_positions[i] * np.sin(doa_rad[j]) / wavelength)
            phase_shift = np.exp(-1j * 2 * np.pi * i * np.sin(doa_rad[j]) / 2)
            rx_signals[i, :] += source_signals[j, :] * phase_shift

    return rx_signals

def add_white_noise(signal, noise_power):

    n_sensors, n_samples = signal.shape

    # Add white noise to the signals
    noise = np.sqrt(noise_power / 2) * (np.random.randn(n_sensors, n_samples) + 1j * np.random.randn(n_sensors, n_samples))
    signal += noise

    return signal