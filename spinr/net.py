import time
import numpy as np
import spinr.tools.cuda
import spinr.forward_numpy
import spinr.forward_cuda
import spinr.module_cuda as module_cuda
import spinr.module_cuda_rate as module_cuda_rate
import spinr.module_cuda_loihi as module_cuda_loihi

class SpiNRNet():
    """
    Spiking Neural Resonator Network
    """

    def __init__(self, n_antennas, 
                        n_distances, n_angles, n_velocities, 
                        params, log=None, kernel='default'):

        self.log = log
        self.kernel = kernel

        # Define input
        self.n_antennas = n_antennas

        # Network architecture
        # Total number of neurons 
        self.n_distances = n_distances
        self.n_angles = n_angles
        self.n_velocities = n_velocities

        # Connections
        self.W = None
        self.sample_R = None
        self.chirp_R = None

        # Neuron parameters
        self.params = params

        # (Optional)
        self.block_shape = None
        self.grid_shape = None

        #
        self.n_samples = None   # defined during runtime
        self.n_chirps = None    # defined during runtime
        self.n_frames = None    # defined during runtime

        self.outs = None
        self.states = None
        self.spikes = None

    def reset(self):

        self.outs = None
        self.states = None
        self.spikes = None
    
    def get_angle_weights(self, n_angles, n_vx):
        """
        Create complex weight matrix for input connections
        Weight matrix values: exp(i*phi) with phi = 2*pi*f_distance*d*cos(theta)*n

        Uniformly spaced phase shifts converted to angles:
        angles = np.arcsin(2*np.linspace(-n_angles//2+1, n_angles//2, n_angles)/n_angles)

        Args:
            n_angles    (int):  Number of output angles
            n_vx        (int):  Number of virtual antennas

        Return:
            (np.array, complex):     Weight matrix
        
        """

        W = np.zeros((n_angles, n_vx)).astype('complex64')
        for a in range(n_angles):
            for rx in range(n_vx):
                phi = 2*np.pi*rx*(a-n_angles//2)/n_angles
                #if phi == 0:
                #    W[a,rx] = 0
                #else:
                W[a,rx] = np.exp(-1j*phi)

        return W

    def get_sample_rotation_weights(self, n_distances, n_samples):
        """
        Create complex rotation vector for smaples.
        Pay attention to type of the input: float or complex.

        Float:      n_distances = n_samples//2
        Complex:    n_distance = n_samples

        Args:
            n_distances (int):  Number of distance neurons

        Return:
            (np.array):     Complex rotation vector
        """

        sample_R = np.exp(1j*np.linspace(0,n_distances-1,n_distances)/(n_samples)*np.pi*2)

        return sample_R

    def get_chirp_rotation_weights(self, n_velocities, n_chirps):
        """
        Create complex rotation vector for chirps.

        Args:
            n_velocities (int):  Number of velocity neurons

        Return:
            (np.array):     Complex rotation vector
        """
        
        chirp_R = np.exp(1j*np.linspace(0,n_velocities-1,n_velocities)/(n_chirps)*np.pi*2)

        return chirp_R

    def init_kernel(self):
        """
        Initialize CUDA kernel.
        """

        if self.kernel=='default':
            mod = module_cuda.init(SourceModule=spinr.tools.cuda.SourceModule)
        elif self.kernel=='rate':
            mod = module_cuda_rate.init(SourceModule=spinr.tools.cuda.SourceModule)
        elif self.kernel=='loihi':
            mod = module_cuda_loihi.init(SourceModule=spinr.tools.cuda.SourceModule)
        kernel_func = mod.get_function("spinr_kernel")

        return kernel_func

    def init_gpu_kernel_allocation(self, n_distances, n_angles, n_velocities):

        # The number of threads per block should be a round multiple
        # of the warp size, which is 32
        # Each block cannot have more than 1024 threads in total
        # Recommendation: Sweet spot between 128 and 512 threads per block
        # The maximum dimensions of each block is limited to [1024,1024,64]
        # The maximum grid dimension is [2^31, 65535, 65535] for 3.x cards
        block_shape = (int(n_velocities), int(n_angles), 1)
        grid_shape = (int(n_distances), 1, 1)

        return block_shape, grid_shape

   
    def init_network_on_gpu(self, debug):
        """
        Initialize network on GPU.
        
        """

        block_shape, grid_shape = self.init_gpu_kernel_allocation(n_distances=self.n_distances, 
                                                                        n_angles=self.n_angles, 
                                                                        n_velocities=self.n_velocities)

        kernel_func = self.init_kernel()

        # Load architecture params on GPU
        W_gpu           = spinr.tools.cuda.complex_to_cuda(self.W.astype('complex64')) 
        sample_R_gpu    = spinr.tools.cuda.complex_to_cuda(self.sample_R.astype('complex64')) 
        chirp_R_gpu     = spinr.tools.cuda.complex_to_cuda(self.chirp_R.astype('complex64')) 
        n_antennas_gpu  = spinr.tools.cuda.mem_alloc_htod(self.n_antennas)

        # Load neuron params on GPU
        params_gpu      = spinr.tools.cuda.mem_alloc_htod(self.params + [debug], dtype=np.float32)

        # Alloc on gpu
        n_samples_gpu   = spinr.tools.cuda.mem_alloc_htod(self.n_samples)
        n_chirps_gpu    = spinr.tools.cuda.mem_alloc_htod(self.n_chirps)
        out_gpu         = spinr.tools.cuda.mem_alloc(self.outs[0].nbytes)

        if debug:
            states_gpu = spinr.tools.cuda.mem_alloc_htod(self.states[0])
            spikes_gpu = spinr.tools.cuda.mem_alloc_htod(self.spikes[0])
        else:
            states_gpu = spinr.tools.cuda.mem_alloc_htod(None)
            spikes_gpu = spinr.tools.cuda.mem_alloc_htod(None)

        return W_gpu, sample_R_gpu, chirp_R_gpu, n_antennas_gpu, params_gpu, \
                    n_samples_gpu, n_chirps_gpu, out_gpu, states_gpu, spikes_gpu, \
                    block_shape, grid_shape, kernel_func


    def init_network_on_cpu(self, debug):

        # Create empty matrix for spikes as numpy output
        if self.kernel=='default':
            # 4 Spiking functions
            self.outs = np.zeros((self.n_frames, self.n_velocities, self.n_distances, self.n_angles, self.n_chirps, 5), dtype=np.float32)
        elif self.kernel=='rate':
            self.outs = np.zeros((self.n_frames, self.n_velocities, self.n_distances, self.n_angles, self.n_chirps, 1), dtype=np.float32)

        # Variables that are shared between neurons
        # 1-dim: gradient
        #self.shared_states = np.zeros((n_frames, self.n_velocities, self.n_distances, self.n_angles), dtype=np.float32)
        #self.shared_states_nbytes = self.shared_states[0].nbytes

        # Create empty matrix for numpy states
        # Number of variables to check can be changed
        if debug:
            if self.kernel=='default':
                self.states = np.zeros((self.n_frames, self.n_velocities, self.n_distances, self.n_angles,
                                        self.n_chirps, self.n_samples, 2), dtype=np.float32)
                self.spikes = np.zeros((self.n_frames, self.n_velocities, self.n_distances, self.n_angles,
                                        self.n_chirps, self.n_samples, 4), dtype=bool)
            elif self.kernel=='rate':
                self.states = np.zeros((self.n_frames, self.n_velocities, self.n_distances, self.n_angles,
                                        self.n_chirps, self.n_samples, 2), dtype=np.float32)
                self.spikes = np.zeros((self.n_frames, self.n_velocities, self.n_distances, self.n_angles,
                                        self.n_chirps, self.n_samples, 1), dtype=bool)

        self.sample_R = self.get_sample_rotation_weights(n_distances=self.n_distances, n_samples=self.n_samples)
        self.chirp_R = self.get_chirp_rotation_weights(n_velocities=self.n_velocities, n_chirps=self.n_chirps)
        self.W = self.get_angle_weights(n_angles=self.n_angles, n_vx=self.n_antennas)


    def forward_gpu(self, input_data, debug=False, devicenum=0):

        self.log.info("")
        self.log.info("SpiNR: Forward_GPU ...")
        self.log.info("")

        function_comp_time = time.time()

        # Runtimes
        mean_comp_time_module = 0 
        memalloc_comp_time = 0
        memcpy_comp_time = 0
        itod_comp_time = 0

        # TODO: Check n_antennas
        # Input shape
        # all time variables are unknown
        # n_antennas can be checked for consistency
        self.n_frames, self.n_chirps, self.n_samples, n_antennas = input_data.shape
        # Input from complex to 2-dim
        input_data = spinr.tools.cuda.complex_to_float(input_data)

        # Init on CPU
        init_net_cpu_comp_time = time.time()
        self.init_network_on_cpu(debug=debug)
        init_net_cpu_comp_time = time.time() - init_net_cpu_comp_time

        # Init GPU
        init_net_gpu_comp_time = time.time()
        spinr.tools.cuda.make_context(devicenum, log=self.log)
        W_gpu, sample_R_gpu, chirp_R_gpu, n_antennas_gpu, params_gpu, \
                    n_samples_gpu, n_chirps_gpu, out_gpu, states_gpu, spikes_gpu, \
                    self.block_shape, self.grid_shape, kernel_func  = self.init_network_on_gpu(debug=debug)
        x_gpu = spinr.tools.cuda.mem_alloc(input_data[0].nbytes)
        init_net_gpu_comp_time = time.time() - init_net_gpu_comp_time

        init_net_comp_time = time.time() - function_comp_time

        for f, x in enumerate(input_data):

            comp_time = time.time()

            # Resetting per frame
            # Needed?
            #if self.debug:
            #    states_gpu = bfrf.tools_cuda2.memcpy_htod(self.states[f])

            # Overwrite allocated variable -> takes longer in total but forward is faster (?)
            # Only re-allocate output data -> faster in total but forward is slower (?)
            # Needed?
            #bfrf.tools_cuda2.memcpy_htod(out_gpu, self.outs[f])
            #Must have
            #shared_states_gpu = bfrf.tools_cuda2.mem_alloc_htod(self.shared_states[f])
            if debug:
                spinr.tools.cuda.memcpy_htod(states_gpu, np.zeros_like(self.states[0]))
                spinr.tools.cuda.memcpy_htod(spikes_gpu, np.zeros_like(self.spikes[0]))

            memalloc_comp_time += time.time() - comp_time

            # Overwrite allocated variable
            comp_time = time.time()
            spinr.tools.cuda.memcpy_htod(x_gpu, x)
            itod_comp_time += time.time() - comp_time

            out_gpu, comp_time_module = spinr.forward_cuda.forward(x_gpu=x_gpu, out_gpu=out_gpu, 
                                                W_gpu=W_gpu, sample_R_gpu=sample_R_gpu, chirp_R_gpu=chirp_R_gpu, 
                                                n_samples_gpu=n_samples_gpu, n_chirps_gpu=n_chirps_gpu, n_antennas_gpu=n_antennas_gpu,
                                                params_gpu=params_gpu, 
                                                states_gpu=states_gpu,
                                                spikes_gpu=spikes_gpu,
                                                #shared_states_gpu=shared_states_gpu,
                                                kernel_func=kernel_func, 
                                                block_shape=self.block_shape, grid_shape=self.grid_shape)

            mean_comp_time_module += comp_time_module

            comp_time = time.time()
            spinr.tools.cuda.memcpy_dtoh(self.outs[f], out_gpu)
            #bfrf.tools_cuda2.memcpy_dtoh(self.shared_states[f], shared_states_gpu)
            if debug:
                spinr.tools.cuda.memcpy_dtoh(self.states[f], states_gpu)
                spinr.tools.cuda.memcpy_dtoh(self.spikes[f], spikes_gpu)

            memcpy_comp_time += time.time() - comp_time

        function_comp_time = time.time() - function_comp_time

        if self.log is not None:
            self.log.info("Init network on CPU (s): "+str(init_net_cpu_comp_time))
            self.log.info("Init network on GPU (s): "+str(init_net_gpu_comp_time))
            self.log.info("Full init network (s): "+str(init_net_comp_time))
            self.log.info("")
            self.log.info("Mean memalloc spikes per sample (s): "+str(memalloc_comp_time/(self.n_frames*self.n_chirps*self.n_samples)))
            self.log.info("Mean input to device per sample (s): "+str(itod_comp_time/(self.n_frames*self.n_chirps*self.n_samples)))
            self.log.info("Mean memcpy DTOH per sample (s) (s): "+str(memcpy_comp_time/(self.n_frames*self.n_chirps*self.n_samples)))
            self.log.info("GPU mean compute time of module per sample (s): "+str(mean_comp_time_module/(self.n_frames*self.n_chirps*self.n_samples)))
            self.log.info("")
            self.log.info("Mean memalloc spikes per frame (s): "+str(memalloc_comp_time/(self.n_frames)))
            self.log.info("Mean input to device per frame (s): "+str(itod_comp_time/(self.n_frames)))
            self.log.info("Mean memcpy DTOH per frame (s) (s): "+str(memcpy_comp_time/(self.n_frames)))
            self.log.info("GPU mean compute time of module per frame (s): "+str(mean_comp_time_module/(self.n_frames)))
            self.log.info("")
            self.log.info("GPU overall compute time: "+str(function_comp_time))

    

        return self.outs

    def forward_cpu(self, input_data, debug=False):

        self.log.info("")
        self.log.info("SpiNR: Forward_CPU ...")
        self.log.info("")

        function_comp_time = time.time()

        # Runtime 
        mean_comp_time_module = 0 

        # TODO: Check n_antennas
        # Input shape
        # all time variables are unknown
        # n_antennas can be checked for consistency
        self.n_frames, self.n_chirps, self.n_samples, n_antennas = input_data.shape

        # Init on CPU
        init_net_cpu_comp_time = time.time()
        self.init_network_on_cpu(debug=debug)
        init_net_cpu_comp_time = time.time() - init_net_cpu_comp_time

        for f, x in enumerate(input_data):
            if debug:
                self.outs[f], comp_time_module = spinr.forward_numpy.forward(x=x, out=self.outs[f], 
                                                W=self.W, sample_R=self.sample_R, chirp_R=self.chirp_R, 
                                                n_samples=self.n_samples, n_chirps=self.n_chirps, n_antennas=self.n_antennas,
                                                n_distances=self.n_distances, n_angles=self.n_angles, n_velocities=self.n_velocities,
                                                params=self.params + [debug], 
                                                states=self.states[f],
                                                spikes=self.spikes[f])
            else:
                self.outs[f], comp_time_module = spinr.forward_numpy.forward(x=x, out=self.outs[f], 
                                                W=self.W, sample_R=self.sample_R, chirp_R=self.chirp_R, 
                                                n_samples=self.n_samples, n_chirps=self.n_chirps, n_antennas=self.n_antennas,
                                                n_distances=self.n_distances, n_angles=self.n_angles, n_velocities=self.n_velocities,
                                                params=self.params + [debug], 
                                                states=self.states,
                                                spikes=self.spikes)

            mean_comp_time_module += comp_time_module

        function_comp_time = time.time() - function_comp_time

        if self.log is not None:
            self.log.info("Init network on CPU (s): "+str(init_net_cpu_comp_time))
            self.log.info("")
            self.log.info("CPU mean compute time of module per frame (s): "+str(mean_comp_time_module/self.n_frames))
            self.log.info("CPU mean compute time of module per sample (s): "+str(mean_comp_time_module/(self.n_frames*self.n_chirps*self.n_samples)))
            self.log.info("")
            self.log.info("CPU overall compute time: "+str(function_comp_time))

    
        return self.outs 



