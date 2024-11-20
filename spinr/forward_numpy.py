import numpy as np
import time
import logging
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)

def forward(x, out, W, sample_R, chirp_R, 
            n_antennas, n_samples, n_chirps,
            n_angles, n_distances, n_velocities,
            states, spikes, params):
    """
    Run network on CPU over single frame but all chirps and samples.

    Args:
        x (cuda):               Radar data input
        out (cuda):             Accumulated spike vector
        W (cuda):               Weight matrix
        sample_R (cuda):        Rotation matrix for samples
        chirp_R (cuda):         Rotation matrix for chirps
        n_samples (cuda):       Number of samples
        n_chirps (cuda):        Number of chirps
        n_antennas (cuda):      Number of virtual antennas
        states (cuda):          State vector over time
        spikes (cuda):          Spike vector over time
        params (cuda):          List of neuron parameters

    Return:
        (np.array):     out
    
    """
    comp_time = time.time()
    W = W.astype('complex64')
    sample_R = sample_R.astype('complex64')
    chirp_R = chirp_R.astype('complex64')
    x = x.astype('complex64')
    out = out.astype('float32')
    if params[-1]:
        states = states.astype('complex64')
        spikes = spikes.astype('complex64')

    # Module

    # Init params
    debug = params[18]
    spiking_function = params[0]
    alpha_x_smooth = params[1]
    alpha_grad_smooth = params[4]
    t_start = params[5]

    alpha_u_rate = params[6]
    weight_u_rate_thresh = params[7]
    offset_u_rate_thresh = params[8]
    weight_u_rate_rest = params[9]
    offset_u_rate_rest = params[10]

    alpha_u_time = params[11]
    weight_u_time_thresh = params[12]
    offset_u_time_thresh = params[13]
    weight_u_time_rest = params[14]
    offset_u_time_rest = params[15]

    weight_delta_thresh = params[16]
    offset_delta_thresh = params[17]

    delta_u_rate_threshold = offset_u_rate_thresh
    u_rate_threshold = offset_u_rate_thresh
    u_rate_rest = offset_u_rate_rest

    u_time_threshold = offset_u_time_thresh
    u_time_rest = offset_u_time_rest

    delta_thresh = offset_delta_thresh

    # Init variables
    grad = np.zeros((n_velocities, n_distances, n_angles)).astype('float32')
    for c in range(n_chirps):
        # Init variables
        smooth_x = np.zeros((n_angles)).astype('complex64')
        s = np.zeros((n_velocities, n_distances, n_angles)).astype('complex64')

        upper = np.zeros((n_velocities, n_distances, n_angles)).astype('float32')
        width = np.zeros((n_velocities, n_distances, n_angles)).astype('float32')

        u_rate = np.zeros((n_velocities, n_distances, n_angles)).astype('float32')
        u_time = np.zeros((n_velocities, n_distances, n_angles)).astype('float32')
        allow_spikes = np.ones((n_velocities, n_distances, n_angles)).astype(bool)
        upper_threshold = np.zeros((n_velocities, n_distances, n_angles)).astype('float32')
        width_threshold = np.zeros((n_velocities, n_distances, n_angles)).astype('float32')

        for t in range(n_samples):

            # Rotate state
            s = np.einsum("j, ijk-> ijk", sample_R, s)

            # Exp filtering
            if t == 0:
                smooth_x = x[c,t]
            smooth_x = (1-alpha_x_smooth) * smooth_x + alpha_x_smooth * x[c,t]

            # Weight matrix
            inp = np.einsum("nk, k-> n", W, (x[c,t] - smooth_x).astype('complex64'))
            inp = np.reshape(inp, (1, 1, n_angles))
            inp = np.repeat(inp, n_velocities, axis=0)
            inp = np.repeat(inp, n_distances, axis=1)
            s += inp

            # Magnitude
            a = np.absolute(s)

            # Envelope
            delta_upper = a-upper
            delta_upper[delta_upper<0] = 0
            delta_width = a-upper-width
            delta_width[delta_width>0] = 0

            # Gradient
            if t >= t_start:
                grad = (1-alpha_grad_smooth)*grad + alpha_grad_smooth*(delta_upper + delta_width)

            # rate LIF
            if spiking_function == 1 or spiking_function == 0:
                if t >= t_start:
                    u_rate += (grad/alpha_grad_smooth + u_rate_rest - u_rate)*alpha_u_rate

                idx = u_rate > u_rate_threshold
                u_rate[idx] -= u_rate_threshold
                out[idx,c,1] += 1

                if debug:
                    spikes[idx,c,0] = True

            # temporal LIF
            if spiking_function == 2 or spiking_function == 0:
                if t >= 0:
                    u_time += (grad/alpha_grad_smooth + u_time_rest - u_time)*alpha_u_time

                idx = (u_time > u_time_threshold) & allow_spikes
                u_time[idx] -= u_time_threshold
                allow_spikes[idx] = False
                out[idx,c,2] = n_samples - t

                if debug:
                    spikes[idx,c,1] = True

            # Adapt Spiking
            if spiking_function == 3 or spiking_function == 0:
                if t == t_start:
                    upper_threshold = upper.copy()
                    width_threshold = width.copy()

                idx = (upper > upper_threshold) & (t>t_start)
                upper_threshold[idx] += delta_thresh
                out[idx,c,3] += 1

                if debug:
                    spikes[idx,c,2] = True

                idx = (width < width_threshold) & (t>t_start)
                width_threshold[idx] -= delta_thresh
                out[idx,c,4] += 1

                if debug:
                    spikes[idx,c,3] = True

            upper += delta_upper
            width += delta_width

            # Output
            if t==n_samples-1:
                out[:,:,:,0,0] = grad

    return out, time.time() - comp_time