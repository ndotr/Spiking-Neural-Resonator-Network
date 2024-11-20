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
        uint spike_dim = 4;
        uint spike_idx = spike_dim*(start_idx_v + start_idx_d + start_idx_a);
        // State data
        // Usage:
        // out_idx + out_dim*n_samples*chirp_idx + out_dim*sample_idx + out_dim_idx
        uint state_dim = 2;
        uint state_idx = state_dim*(start_idx_v + start_idx_d + start_idx_a);
        // Output data
        uint out_dim = 5;
        uint out_idx = n_chirps * out_dim * (n_angles*n_distances*velocity_idx + n_angles*distance_idx + angle_idx);
        uint out1 = 0;
        uint out2 = 0;
        uint out3 = 0;
        uint out4 = 0;
                          

        // W
        // Usage: 
        // start_idx_W + 2*antenna_idx + 0/1 (real/imag)
        uint start_idx_W = 2*n_antennas*angle_idx;

        // Parameters
        // Global params 
        //uint n_epochs = params[12];
        bool debug = params[18];
        uint spiking_function = params[0];
        //bool bool_share_weights = params[1]; // boolean whether weights are shared between a range population

        // Input params
        float alpha_x_smooth = params[1]; // exponential filtering of input x, default: 0.01 or 0
        float alpha_a_smooth = params[2]; // exponential filtering of modulus a, default: 0.01
        float alpha_l = params[3]; // weight of negative gradient / lower limit, default: 1
        float alpha_grad_smooth = params[4]; // exponential filtering of gradient, default: 0.001

        // Start time for gradient estimation
        uint t_start = params[5];

        // Spiking Function Params

        // LIF for spike generation with potential u
        float alpha_u_rate = params[6]; // weight for potential, default: 0
        // u thresh
        // weight * mean_grad + offset for LIF threshold
        float weight_u_rate_thresh = params[7]; 
        float offset_u_rate_thresh = params[8];
        // u rest
        // weight * mean_grad + offset for LIF resting potential
        float weight_u_rate_rest = params[9];
        float offset_u_rate_rest = params[10];

        // Temporal LIF for spike generation with potential u
        float alpha_u_time = params[11]; // weight for potential, default: 0
        // u thresh
        // weight * mean_grad + offset for LIF threshold
        float weight_u_time_thresh = params[12]; 
        float offset_u_time_thresh = params[13];
        // u rest
        // weight * mean_grad + offset for LIF resting potential
        float weight_u_time_rest = params[14];
        float offset_u_time_rest = params[15];
                          
        // Adaptive threshold for spike generation
        // delta thresh
        // weight * mean_grad + offset for stepsize of threshold
        float weight_delta_thresh = params[16]; 
        float offset_delta_thresh = params[17]; 

        // If bool_share_weights == False, set fixed threshold and rest for LIF                  
        float delta_u_rate_threshold = offset_u_rate_thresh;
        float u_rate_threshold = offset_u_rate_thresh;
        float u_rate_rest = offset_u_rate_rest;
        // If bool_share_weights == False, set fixed threshold and rest for temporal LIF                  
        float u_time_threshold = offset_u_time_thresh;
        float u_time_rest = offset_u_time_rest;
        // If bool_share_weights == False, set fixed delta thresh
        float delta_thresh = offset_delta_thresh;



        // Init dynamic states
        // Input states
        //float w[2] = {1,0};
        //float w_[2] = {1,0};
        float inp[2] = {0,0};
        //float pass_x[64] = {0};
        //float pass_x_[64] = {0};
        float smooth_x[64] = {0};
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
        float u_rate = 0;
        float u_time = 0;
        // Reset refractory period for each chirp 
        bool allow_spike = true;

        // Adaptive thresholds
        float upper_threshold = 0;
        float lower_threshold = 0;


        for (int c=0; c<n_chirps; c++) {
            
            out1 = 0;
            out2 = 0;
            out3 = 0;
            out4 = 0;

            // RF Neuron states
            float s[2] = {0, 0};
            float s_[2] = {0,0};

            // Reset filtered input for each chirp
            smooth_x[0] = 0;
            smooth_x[1] = 0;

            // Reset gradient of neuron state
            grad = 0;
             
            // Reset envelope
            lower = 0;
            upper = 0;

            // Reset adaptive thresholds
            upper_threshold = 0;
            lower_threshold = 0;

            // Reset LIF
            u_rate = 0;
            u_time = 0;
            // Reset refractory period for each chirp 
            allow_spike = true;

            // Init states
            //s[0] = chirp_R[2*n_velocities*distance_idx + 2*velocity_idx + 0] * s_[0] - chirp_R[2*n_velocities*distance_idx + 2*velocity_idx + 1] * s_[1];
            //s[1] = chirp_R[2*n_velocities*distance_idx + 2*velocity_idx + 0] * s_[1] + chirp_R[2*n_velocities*distance_idx + 2*velocity_idx + 1] * s_[0];
            if (n_velocities == 1){
                s[0] = 0;
                s[1] = 0;
                
                // Reset envelope
                lower = 0;
                upper = 0;
            }
            else {
                s[0] = chirp_R[2*velocity_idx + 0] * s_[0] - chirp_R[2*velocity_idx + 1] * s_[1];
                s[1] = chirp_R[2*velocity_idx + 0] * s_[1] + chirp_R[2*velocity_idx + 1] * s_[0];
            }
            //w[0] = chirp_R[2*velocity_idx + 0] * w_[0] - chirp_R[2*velocity_idx + 1] * w_[1];
            //w[1] = chirp_R[2*velocity_idx + 0] * w_[1] + chirp_R[2*velocity_idx + 1] * w_[0];

        for (int t=0; t<n_samples; t++) {

            // Variables for matrix multiplication inp = W x_t
            inp[0] = 0;
            inp[1] = 0;

            // Rotation of states s_t+1 = R s_t
            // with rotation matrix R
            s_[0] = 1*(sample_R[2*distance_idx + 0] * s[0] - sample_R[2*distance_idx + 1] * s[1]);
            s_[1] = 1*(sample_R[2*distance_idx + 1] * s[0] + sample_R[2*distance_idx + 0] * s[1]);

            // Optimize it
            //for (int i=0; i<n_antennas; i++) {
            //    if (t>=0){
            //    smooth_x_[0] = ((1-alpha_x_smooth)*smooth_x_[0] + alpha_x_smooth*x[2*n_antennas*n_samples*c + 2*n_antennas*t + 2*i + 0]);
            //    smooth_x_[1] = ((1-alpha_x_smooth)*smooth_x_[1] + alpha_x_smooth*x[2*n_antennas*n_samples*c + 2*n_antennas*t + 2*i + 1]);
            //    }
            //    else {
            //        smooth_x_[0] += x[2*n_antennas*n_samples*c + 2*n_antennas*t + 2*i + 0]/n_antennas;
            //        smooth_x_[1] += x[2*n_antennas*n_samples*c + 2*n_antennas*t + 2*i + 1]/n_antennas;
            //    }
            //}

            // Matrix multiplication W x_t
            for (int i=0; i<n_antennas; i++) {
                
                //pass_x[2*i+0] += x[2*n_antennas*n_samples*c + 2*n_antennas*t + 2*i + 0]; //- smooth_x_[0];
                //pass_x[2*i+1] += x[2*n_antennas*n_samples*c + 2*n_antennas*t + 2*i + 1]; //- smooth_x_[1];
                //pass_x_[2*i+0] = 0.95*(sample_R[2*distance_idx + 0] * pass_x[2*i+0] - sample_R[2*distance_idx + 1] * pass_x[2*i+1]);
                //pass_x_[2*i+1] = 0.95*(sample_R[2*distance_idx + 1] * pass_x[2*i+0] + sample_R[2*distance_idx + 0] * pass_x[2*i+1]);
                //pass_x[2*i+0] = pass_x_[2*i+0]; ///sqrtf(pass_x_[2*i+0]*pass_x_[2*i+0] + pass_x_[2*i+1]*pass_x_[2*i+1]);
                //pass_x[2*i+1] = pass_x_[2*i+1]; ///sqrtf(pass_x_[2*i+0]*pass_x_[2*i+0] + pass_x_[2*i+1]*pass_x_[2*i+1]);
                
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
                          

                //inp[0] += W[start_idx_W + 2*i + 0] * (pass_x[2*i+0]/sqrtf(pass_x[2*i+0]*pass_x[2*i+0] + pass_x[2*i+1]*pass_x[2*i+1]))
                //              - W[start_idx_W + 2*i + 1] * (pass_x[2*i+1]/sqrtf(pass_x[2*i+0]*pass_x[2*i+0] + pass_x[2*i+1]*pass_x[2*i+1]));
                //inp[1] += W[start_idx_W + 2*i + 1] * (pass_x[2*i+0]/sqrtf(pass_x[2*i+0]*pass_x[2*i+0] + pass_x[2*i+1]*pass_x[2*i+1]))
                //               + W[start_idx_W + 2*i + 0] * (pass_x[2*i+1]/sqrtf(pass_x[2*i+0]*pass_x[2*i+0] + pass_x[2*i+1]*pass_x[2*i+1]));

                //inp[0] += W[start_idx_W + 2*i + 0] * (pass_x[2*i+0])
                //              - W[start_idx_W + 2*i + 1] * (pass_x[2*i+1]);
                //inp[1] += W[start_idx_W + 2*i + 1] * (pass_x[2*i+0])
                //               + W[start_idx_W + 2*i + 0] * (pass_x[2*i+1]);

                //inp[0] += W[start_idx_W + 2*i + 0] * (pass_x[2*i+0])
                //              - W[start_idx_W + 2*i + 1] * (pass_x[2*i+1]);
                //inp[1] += W[start_idx_W + 2*i + 1] * (pass_x[2*i+0])
                //               + W[start_idx_W + 2*i + 0] * (pass_x[2*i+1]);


            }

            // Adding s_t + Wx_t
            s_[0] += inp[0];
            s_[1] += inp[1];
                          
            a = sqrtf(s_[0]*s_[0] + s_[1]*s_[1]);
            a = (1-alpha_a_smooth)*a + alpha_a_smooth * (abs(s_[0]) + abs(s_[1]));
            
            // Envelopes
            delta_upper =  1*(a-upper)*((upper)<(a));
            delta_lower = ((upper + lower) > a)*(a-lower-upper)*alpha_l;

            // Sharing weights 
            //if (t==1){
            //    if (bool_share_weights){
            //        mean_grad = 0;
            //        for (int i=0; i<n_angles; i++) {
            //            //if ((abs(i-int(angle_idx)) > 0) && (abs(i-int(angle_idx)) <= 3)){
            //            //    cfar_grad += shared_states[n_angles*range_idx + i];
            //            //}
            //            mean_grad += shared_states[n_angles*range_idx + i];
            //        }
            //        //for (int i=0; i<n_angles; i++) {
            //        //    std_grad += abs(mean_grad/n_angles - shared_weights[n_angles*range_idx + i]);
            //        //}
            //        //u_grad_threshold = (cfar_grad)*1.5+offset_u_grad_tresh;

            //        u_grad_threshold = weight_u_grad_thresh*mean_grad/n_angles+offset_u_grad_thresh;
            //        u_rest = weight_u_rest*mean_grad+offset_u_rest;
            //        u_grad_threshold_temporal = weight_u_grad_thresh_temporal*mean_grad/n_angles+offset_u_grad_thresh_temporal;
            //        u_rest_temporal = weight_u_rest_temporal*mean_grad+offset_u_rest_temporal;
            //        delta_thresh = mean_grad*weight_delta_thresh + offset_delta_thresh;
            //    }
            //}

            // Exponential filtering
            // Estimate gradient with starting point
            if ((t>=t_start)) {
                //grad += -alpha_grad_smooth*grad + alpha_grad_smooth*(delta_upper+delta_lower);
                grad =  (1-alpha_grad_smooth)*grad + alpha_grad_smooth*(delta_upper+delta_lower);
            }

            if ((spiking_function == 1) || (spiking_function == 0)) {
            // LIF dynamics
            if ((t>=t_start)) {
                // LIF
                u_rate += (grad/alpha_grad_smooth + u_rate_rest - u_rate)*alpha_u_rate;
                //u_rate += (grad + u_rate_rest - u_rate)*alpha_u_rate;
                // IF
                //u_rate += grad/alpha_grad_smooth + u_rate_rest;
            }
            // Spiking block
            // creates spikes even before t_start, using previous u_grad
            if ((u_rate>u_rate_threshold)){
                u_rate -= u_rate_threshold;
                //u_rate_threshold += delta_u_rate_threshold;
                out1 += 1;

                if (debug==true)  {
                    spikes[spike_idx + spike_dim*n_samples*c + spike_dim*t + 0] = true;
                }
            }
            }

            if ((spiking_function == 2) || (spiking_function == 0)) {
            // Temporal LIF dynamics
            if ((t>0)) {
                u_time += (grad/alpha_grad_smooth + u_time_rest - u_time)*alpha_u_time;
                //u_time += grad/alpha_grad_smooth + u_time_rest;
            }
            // Temporal Spiking block
            // creates spikes even before t_start, using previous u_grad
            if ((u_time>u_time_threshold) && (allow_spike)){
                allow_spike = false;
                out2 = n_samples - t;

                if (debug==true) {
                    spikes[spike_idx + spike_dim*n_samples*c + spike_dim*t + 1] = true;
                }
            }
            }

            if ((spiking_function == 3) || (spiking_function == 0)) {
            // Adapt spiking
            // Set starting point for spikes
            if ((t==t_start)){
                upper_threshold = upper;
                lower_threshold = lower;
            }
            // Pos Spiking block
            if ((upper  > upper_threshold) && (t>t_start)){ //&& (sum_width_grad > 0.)){
                upper_threshold += delta_thresh;
                out3 += 1;

                if (debug==true) {
                    spikes[spike_idx + spike_dim*n_samples*c + spike_dim*t + 2] = true;
                }
            }
            // Neg Spiking block
            if ((lower  < lower_threshold) && (t>t_start)){ //&& (sum_width_grad > 0.)){
                lower_threshold -= delta_thresh;
                out4 += 1;

                if (debug==true) {
                    spikes[spike_idx + spike_dim*n_samples*c + spike_dim*t + 3] = true;
                }
            }
            }
                          
            // Updating 
            upper += delta_upper;
            lower += delta_lower;
            s[0] = s_[0];
            s[1] = s_[1];

            // Storing 
            if (debug==true) {
                states[state_idx + state_dim*n_samples*c + state_dim*t + 0] = s_[0];
                states[state_idx + state_dim*n_samples*c + state_dim*t + 1] = s_[1];
            }
            //if ((t == (n_samples-1)) && (c==(n_chirps-1))){
            if ((t == (n_samples-1))){
                out[out_idx + out_dim*c + 0] = grad * (grad>0);
                out[out_idx + out_dim*c + 1] = out1;
                out[out_idx + out_dim*c + 2] = out2;
                out[out_idx + out_dim*c + 3] = out3;
                out[out_idx + out_dim*c + 4] = out4;
            }
        }
        //w[0] = w_[0];
        //w[1] = w_[1];
        }
    }
    //!cuda
    """)

    return rf_mod

