# vim: filetype=pycuda.python
# vim: filetype=pycuda
def init(SourceModule):

    rf_mod = SourceModule("""//CUDA//
    //cuda
    #include <cuda.h>
    #include<stdio.h>
    #include<stdlib.h>
    __global__ void spinr_kernel(float *x, float *out, float *W, float *sample_R, float *chirp_R, 
                          int* n_samples_, int* n_chirps_, int* n_antennas_,
                          float *states,
                          bool *spikes, 
                          //float *shared_states, 
                          float *params)
    {
        uint angle_idx = threadIdx.y;
        uint distance_idx = blockIdx.x;
        uint velocity_idx = threadIdx.x;
        
        uint n_distances = gridDim.x;
        uint n_velocities = blockDim.x;
        uint n_angles = blockDim.y;

        uint n_samples = n_samples_[0];
        uint n_antennas = n_antennas_[0];
        uint n_chirps = n_chirps_[0];
                          
        uint start_idx_v = n_samples*n_chirps*n_angles*n_distances*velocity_idx;
        uint start_idx_d = n_samples*n_chirps*n_angles*distance_idx;
        uint start_idx_a = n_samples*n_chirps*angle_idx;

        // Spike data
        // Usage:
        // spike_idx + spike_dim*n_samples*chirp_idx + spike_dim*sample_idx + spike_dim_idx
        uint spike_dim = 1;
        uint spike_idx = spike_dim*(start_idx_v + start_idx_d + start_idx_a);
        // State data
        // Usage:
        // out_idx + out_dim*n_samples*chirp_idx + out_dim*sample_idx + out_dim_idx
        uint state_dim = 2;
        uint state_idx = state_dim*(start_idx_v + start_idx_d + start_idx_a);
        // Output data
        uint out_dim = 1;
        uint out_idx = n_chirps * out_dim * (n_angles*n_distances*velocity_idx + n_angles*distance_idx + angle_idx);
        uint out0 = 0;

        // W
        // Usage: 
        // start_idx_W + 2*antenna_idx + 0/1 (real/imag)
        uint start_idx_W = 2*n_antennas*angle_idx;

        // Parameters
        // Global params 
        bool debug = params[8];

        // Input params
        float alpha_x_smooth = params[0]; // exponential filtering of input x, default: 0.6
        float alpha_l = params[1]; // weight of negative gradient / lower limit, default: 1
        float alpha_a_smooth = params[2]; // exponential filtering of magnitude, default: 0.001
        float alpha_grad_smooth = params[3]; // exponential filtering of gradient, default: 0.001

        // Start time for gradient estimation
        uint t_start = params[4];

        // Spiking Function Params
        // LIF for spike generation with potential u
        float alpha_u_time = params[5];     // weight for potential, default: 0
        float u_time_threshold = params[6]; // threshold
        float u_time_rest = params[7];      // rest

        // Init dynamic states
        // Input states
        float inp[2] = {0,0};
        float smooth_x_[64] = {0};
            
        // RF Neuron states
        float s[2] = {0, 0};
        float s_[2] = {0,0};

        // Magnitude
        float a = 0;
        
        // Envelopes
        float lower = 0;
        float upper = 0;
        float delta_lower = 0;
        float delta_upper = 0;

        // Gradient
        float grad = 0;

        // Rate and time LIF
        float u_time = 0;
        bool allow_spike = true;

        for (int c=0; c<n_chirps; c++) {
            
            out0 = 0;

            // Reset filtered input for each chirp
            //smooth_x_[64] = {0};
            //memset(smooth_x_, 0, sizeof(smooth_x_));
             
            // Reset LIF
            u_rate = 0;

            // Init states
            if (n_velocities == 1){
                //s[0] = 0;
                //s[1] = 0;
                
                // RF Neuron states
                //s[2] = {0, 0};
                //s_[2] = {0,0};
                memset(s, 0, sizeof(s));
                memset(s_, 0, sizeof(s_));

                // Reset envelope
                lower = 0;
                upper = 0;

                // Reset gradient of neuron state
                grad = 0;
                allow_spike = true;
            }
            else {
                s[0] = chirp_R[2*velocity_idx + 0] * s_[0] - chirp_R[2*velocity_idx + 1] * s_[1];
                s[1] = chirp_R[2*velocity_idx + 0] * s_[1] + chirp_R[2*velocity_idx + 1] * s_[0];
                // Reset envelope
                lower = 0;
                upper = 0;
                allow_spike = true;
            }

        for (int t=0; t<n_samples; t++) {

            // Variables for matrix multiplication inp = W x_t
            inp[0] = 0;
            inp[1] = 0;

            // Rotation of states s_t+1 = R s_t
            // with rotation matrix R
            s_[0] = 1*(sample_R[2*distance_idx + 0] * s[0] - sample_R[2*distance_idx + 1] * s[1]);
            s_[1] = 1*(sample_R[2*distance_idx + 1] * s[0] + sample_R[2*distance_idx + 0] * s[1]);


            // Matrix multiplication W x_t
            for (int i=0; i<n_antennas; i++) {
                
                if (t==0){
                    smooth_x_[2*i+0] = x[2*n_antennas*n_samples*c + 2*n_antennas*t + 2*i + 0];
                    smooth_x_[2*i+1] = x[2*n_antennas*n_samples*c + 2*n_antennas*t + 2*i + 1];

                }
                else {
                    smooth_x_[2*i+0] = (1-alpha_x_smooth)*smooth_x_[2*i+0] + alpha_x_smooth*x[2*n_antennas*n_samples*c + 2*n_antennas*t + 2*i + 0];
                    smooth_x_[2*i+1] = (1-alpha_x_smooth)*smooth_x_[2*i+1] + alpha_x_smooth*x[2*n_antennas*n_samples*c + 2*n_antennas*t + 2*i + 1];
                }
                          

                inp[0] += W[start_idx_W + 2*i + 0] * (x[2*n_antennas*n_samples*c + 2*n_antennas*t + 2*i + 0] - smooth_x_[2*i+0])
                              - W[start_idx_W + 2*i + 1] * (x[2*n_antennas*n_samples*c + 2*n_antennas*t + 2*i + 1] - smooth_x_[2*i+1]);
                inp[1] += W[start_idx_W + 2*i + 1] * (x[2*n_antennas*n_samples*c + 2*n_antennas*t + 2*i + 0] - smooth_x_[2*i+0])
                               + W[start_idx_W + 2*i + 0] * (x[2*n_antennas*n_samples*c + 2*n_antennas*t + 2*i + 1] - smooth_x_[2*i+1]);
                          
            }

            // Adding s_t + Wx_t
            s_[0] += inp[0];
            s_[1] += inp[1];
                          
            //a = sqrtf(s_[0]*s_[0] + s_[1]*s_[1]);
            a = (1-alpha_a_smooth)*a + alpha_a_smooth * (abs(s_[0]) + abs(s_[1]));
            
            // Envelopes
            delta_upper =  1*(a-upper)*((upper)<(a));
            delta_lower = ((upper + lower) > a)*(a-lower-upper)*alpha_l;

            // Exponential filtering
            // Estimate gradient with starting point
            if ((t>=t_start)) {
                grad =  (1-alpha_grad_smooth)*grad + alpha_grad_smooth*(delta_upper+delta_lower);
                u_time += (grad/alpha_grad_smooth + u_time_rest - u_time)*alpha_u_time;
            }

            // Spiking block
            if ((u_time>u_time_threshold) && (allow_spike)){
                allow_spike = false;
                out0 = n_samples - t;

                if (debug==true) {
                    spikes[spike_idx + spike_dim*n_samples*c + spike_dim*t + 1] = true;
                }
            }

                          
            // Updating 
            upper += delta_upper;
            lower += delta_lower;
            s[0] = s_[0];
            s[1] = s_[1];

            // Storing 
            if (debug==true) {
                states[state_idx + state_dim*n_samples*c + state_dim*t + 0] = u_time;
                states[state_idx + state_dim*n_samples*c + state_dim*t + 1] = grad;
            }
            if ((t == (n_samples-1)) && (c == (n_chirps-1))){
                out[out_idx + out_dim*c + 0] = out0;
            }
        }
        }
        out[out_idx + out_dim*(n_chirps-1) + 0] = out0;
    }
    //!cuda
    """)

    return rf_mod

