import time
import numpy as np
import spinr.tools.cuda

#def prepare_network_architecture_on_gpu(W, sample_R, chirp_R, n_antennas, params):
#    """
#    Transform and load network parameters to GPU.
#
#    Args:
#        W (np.array):           static weight matrix
#        sample_R (np.array):    rotation matrix for sample rotation
#        chirp_R (np.array):     rotation matrix for chirp rotation
#        n_antennas (int):       number of inputs / number of virtual antennas
#        params (list):          neuron dynamic and spiking function parameters       
#
#    Return:
#        cuda_object:    weight matrix on gpu
#        cuda_object:    rotation matrix for samples on gpu
#        cuda_object:    rotation matrix for chirps on gpu
#        cuda_object:    number of virtual antennas on gpu
#        cuda_object:    params on gpu
#    
#    """
#
#    # Load architecture params on GPU
#    W_gpu           = spinr.tools.cuda.complex_to_cuda(W.astype('complex64')) 
#    sample_R_gpu    = spinr.tools.cuda.complex_to_cuda(sample_R.astype('complex64')) 
#    chirp_R_gpu     = spinr.tools.cuda.complex_to_cuda(chirp_R.astype('complex64')) 
#    n_antennas_gpu  = spinr.tools.cuda.mem_alloc_htod(n_antennas)
#
#    # Load neuron params on GPU
#    params_gpu      = spinr.tools.cuda.mem_alloc_htod(params, dtype=np.float32)
#
#    return W_gpu, sample_R_gpu, chirp_R_gpu, n_antennas_gpu, params_gpu


def forward(x_gpu, out_gpu, 
            W_gpu, sample_R_gpu, chirp_R_gpu,
            n_samples_gpu, n_chirps_gpu, n_antennas_gpu,
            states_gpu, 
            spikes_gpu, 
            #shared_states_gpu,
            params_gpu, 
            block_shape, grid_shape, kernel_func):
    """
    Run network on CUDA over single frame but all chirps and samples.

    Args:
        x_gpu (cuda):               Radar data input
        out_gpu (cuda):             Accumulated spike vector
        W_gpu (cuda):               Weight matrix
        sample_R_gpu (cuda):        Rotation matrix for samples
        chirp_R_gpu (cuda):         Rotation matrix for chirps
        n_samples_gpu (cuda):       Number of samples
        n_chirps_gpu (cuda):        Number of chirps
        n_antennas_gpu (cuda):      Number of virtual antennas
        states_gpu (cuda):          State vector over time
        spikes_gpu (cuda):          Spike vector over time
        params_gpu (cuda):          List of neuron parameters
        block_shape (tuple of int): Block size
        grid_shape (tuple of int):  Grid size
        kernel_func (CudaModule):   Neuron dynamics

    Return:
        (cuda_object):  Spike vector
        float:          Kernel runtime 

    
    """

    if not spinr.tools.cuda.cuda:
        print("No PyCuda functionality.")
        return -1

    # Time runtime
    comp_time = time.time()

    # Run CUDA kernel 
    kernel_func(x_gpu, out_gpu, 
                W_gpu, sample_R_gpu, chirp_R_gpu,
                n_samples_gpu, n_chirps_gpu, n_antennas_gpu,
                states_gpu, 
                spikes_gpu,
                params_gpu,
                block=block_shape, grid=grid_shape)
    
    spinr.tools.cuda.cuda.Context.synchronize()

    comp_time = time.time() - comp_time

    return out_gpu, comp_time





