#!/bin/bash
# Basic command to run the script for MNIST dataset with sigma = 15.0
# Arguments are in order:
# beta, lr, sigma, batch_size, P, seed, feature_type

python Conv_relus_MNIST.py 0.01 0.00031 15.0 1024 100 42 0
