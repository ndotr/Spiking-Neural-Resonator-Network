###
# Configuration file
#
# Share configurations between docker and classic runs.
#
###

import datetime
import numpy as np

# CUDA device
devicenum = 0

# Network architecture
n_chirps = 8
n_samples = 512

# Output data
remove_zero_modes = 4

# Neuron parameters
temporal_threshold = 0.35 #0.35 # default: 0.0005
rate_threshold = 40 #35 # default: 0.0005
params = [          # debug
            0,      # spiking function (0: all, 1:, rate LIF, 2: temporal LIF, 3: adapt threshold)
            # Input
            0.6,   # alpha x smooth, default: 0.0005
            0,   # alpha a smooth 0.01
            # Grad
            1,      # alpha lower, default: 1
            0.0001,  # alpha grad smooth, default: 0.001
            32,     # t start, default: 32
            # Rate LIF
            0.01,  # alpha u grad, default: 0.0
            0,      # weight u grad thresh
            rate_threshold*0.01,    # default: 35. offset u grad thresh, default: 0.125 = THRESHOLD
            0,      # weight u rest
            0,      # offset u rest
            # Temporal LIF
            0.005,  # default: 0.00001. alpha u grad, default: 0.005
            0,      # weight u grad thresh
            (250+temporal_threshold)*(1-np.exp(-512*0.005)),    # default: 2. offset u grad thresh, default: 1.
            0,      # weight u rest
            250,      # offset u rest
            # Adapt thresh
            0,      # weight delta thresh
            0.1,    # default: 0.2, offset delta thresh = THRESHOLD
          ]

