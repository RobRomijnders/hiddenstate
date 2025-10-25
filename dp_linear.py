import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from opacus.data_loader import DPDataLoader

from torch.func import functional_call,  grad
import torch.nn.functional as F
import torch.nn as nn

import Conv_relus_MNIST as cvxnn


def generate_relu_features(RAW_train, RAW_test, num_neurons, do_extend=False):
   # Training data
    num_neurons, sign_pattern_list, u_vector_list = cvxnn.generate_sign_patterns(RAW_train, num_neurons, verbose=True)
    sign_patterns = torch.stack(sign_pattern_list).int().transpose(0, 1).numpy()

    u_vectors = torch.Tensor(u_vector_list).reshape((num_neurons, RAW_train.shape[1])).T
    sign_patterns_test = (RAW_test @ u_vectors) >= 0

    # Convert to numpy arrays for linear regression
    X_relu_train = torch.Tensor(sign_patterns).float() # Shape: (n_samples, n_features)
    if do_extend:
        X_relu_train = (X_relu_train.unsqueeze(-1) * RAW_train.unsqueeze(1)).reshape(X_relu_train.shape[0], -1)
    X_relu_train = torch.cat([X_relu_train, torch.ones(X_relu_train.shape[0], 1)], dim=1)

    X_relu_test = torch.Tensor(sign_patterns_test).float()
    if do_extend:
        X_relu_test = (X_relu_test.unsqueeze(-1) * RAW_test.unsqueeze(1)).reshape(X_relu_test.shape[0], -1)
    X_relu_test = torch.cat([X_relu_test, torch.ones(X_relu_test.shape[0], 1)], dim=1)

    return X_relu_train, X_relu_test

# Implement Random Fourier Features (RFF)
def generate_rff(X, n_components, gamma=1.0, do_extend=False):
    """
    Generate Random Fourier Features for approximating RBF kernel
    X: Input data (n_samples, n_features)
    n_components: Number of random features to generate
    gamma: RBF kernel parameter
    """
    _, n_features = X.shape

    # Sample random weights from Normal distribution
    W = torch.sqrt(2 * gamma) * torch.randn(n_features, n_components)
    b = 2 * torch.pi * torch.rand(n_components)

    # Project data
    Z = torch.sqrt(torch.tensor(2/n_components)) * torch.cos(X @ W + b)

    if do_extend:
        Z = (Z.unsqueeze(-1) * X.unsqueeze(1)).reshape(X.shape[0], -1)

    return Z, W, b


def generate_rff_test(X, W, b, n_components, do_extend=False):
    Z = np.sqrt(2/n_components) * np.cos(X @ W + b)

    if do_extend:
        Z = (Z.unsqueeze(-1) * X.unsqueeze(1)).reshape(X.shape[0], -1)
    return Z


@torch.compile
def priv_linear_regression(X, y, beta, clip, epsilon=1.0, delta=1e-6, num_classes=10):
  device = X.device
  with torch.no_grad():
    _, num_features = X.shape
    norms = torch.linalg.norm(X, axis=1)
    X_clipped = X * torch.maximum(torch.tensor(1.0), clip / norms[:, None])

    # Write out the sensitivity of the denominator, related to the smallest eigenvalue of X.T @ X?
    # Perhaps we can do a trick because we know for random fourier fetures that all values are between -1 and 1?
    # Multiply by two as we spend half epsilon on numerator and half epsilon on denominator
    # Multiply by num_classes as we have num_classes classes, could be (num_classes - 1) if we want to be more careful?
    sigma_numerator = num_classes * 2 * (clip / epsilon) * np.sqrt(2 * np.log(1.25 / delta))
    sigma_denominator = 2 * (clip**2 / epsilon) * np.sqrt(2 * np.log(1.25 / delta))  # TODO: sensitivity is C**2 because it's a matrix?

    denominator = X_clipped.T @ X_clipped + beta * torch.eye(num_features).to(device) + sigma_denominator * torch.randn(num_features, num_features).to(device)
    numerator = X_clipped.T @ y + sigma_numerator * torch.randn(num_features, num_classes).to(device)
    return torch.linalg.inv(denominator) @ numerator


@torch.compile
def adassp(features, labels, epsilon, delta, rho=0.05, num_classes=10):
  """Returns model computed using AdaSSP DP linear regression.

  Re-use from https://github.com/google-research/google-research/blob/master/dp_regression/baselines.py#L48

  Args:
    features: Matrix of feature vectors. Assumed to have intercept feature.
    labels: Vector of labels.
    epsilon: Computed model satisfies (epsilon, delta)-DP.
    delta: Computed model satisfies (epsilon, delta)-DP.
    rho: Failure probability. The default of 0.05 is the one used in
      https://arxiv.org/pdf/1803.02596.pdf.

  Returns:
    Vector of regression coefficients. AdaSSP is described in Algorithm 2 of
    https://arxiv.org/pdf/1803.02596.pdf.
  """
  device = features.device
  with torch.no_grad():
    _, d = features.shape
    # these bounds are data-dependent and not dp
    bound_x = torch.max(torch.linalg.norm(features, axis=1))
    bound_y = 1.0

    covar_xx = torch.matmul(features.T, features)

    lambda_min = max(0, torch.amin(torch.abs(torch.linalg.eigvals(covar_xx))))
    z = torch.randn(1).float().to(device)  # np.random.normal(size=1)
    sensitivity = torch.tensor(np.sqrt(np.log(6 / delta)) / (epsilon / 3)).float().to(device)
    private_lambda = max(
        0, lambda_min + sensitivity * (bound_x**2) * z -
        (bound_x**2) * np.log(6 / delta) / (epsilon / 3))
    final_lambda = max(
      0, torch.tensor(np.sqrt(d * np.log(6 / delta) * np.log(2 * (d**2) / rho))).float().to(device)
      * (bound_x**2) / (epsilon / 3) - private_lambda)

    # generate symmetric noise_matrix where each upper entry is iid N(0,1)
    noise_matrix = torch.randn(d, d).float().to(device)
    noise_matrix = torch.triu(noise_matrix)
    noise_matrix = noise_matrix + noise_matrix.T - torch.diag(torch.diag(noise_matrix))

    priv_xx = covar_xx + sensitivity * (bound_x**2) * noise_matrix
    priv_xy = torch.matmul(features.T, labels) + num_classes * sensitivity * bound_x * bound_y * torch.randn(d, num_classes).float().to(device)

    eye_matrix = final_lambda * torch.eye(d).float().to(device)
    # model_adassp = torch.matmul(torch.linalg.pinv(priv_xx + eye_matrix), priv_xy)
    return torch.linalg.solve(priv_xx + eye_matrix, priv_xy)
