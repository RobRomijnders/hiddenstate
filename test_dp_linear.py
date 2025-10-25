import dp_linear
import torch
import numpy as np

def test_generate_rff():
    X = torch.randn(100, 10)
    n_components = 10
    gamma = torch.tensor(1.0)
    do_extend = False
    Z, W, b = dp_linear.generate_rff(X, n_components, gamma, do_extend)
    assert Z.shape == (100, n_components)

    Z_test = dp_linear.generate_rff_test(X, W, b, n_components, do_extend)
    assert Z_test.shape == (100, n_components)

    Z_test_2 = dp_linear.generate_rff_test(X, W, b, n_components, do_extend)
    assert torch.allclose(Z_test, Z_test_2)


def test_linear_regression():
    y = torch.randn(100, 1)

    X = torch.hstack([y, torch.ones(100, 1), 0.1 * torch.randn(100, 3)])
    beta = torch.tensor(0.000001)
    clip = torch.tensor(100.0)
    epsilon = torch.tensor(1000.0)
    delta = torch.tensor(0.5)

    weights = dp_linear.priv_linear_regression(X, y, beta, clip, epsilon, delta, num_classes=1)
    weights = weights.detach().numpy()
    assert weights.shape == (5, 1), f"weights.shape: {weights.shape}"
    # Test is such that the first feature is the most predictive
    assert weights[0] > np.sum(weights[1:]), f"weights[0]: {weights[0]}, sum(weights[1:]): {np.sum(weights[1:])}"


def test_adassp_eps1000():
    y = torch.randn(100, 1)

    X = torch.hstack([y, torch.ones(100, 1), 0.1 * torch.randn(100, 3)])
    epsilon = torch.tensor(1000.0)
    delta = torch.tensor(0.5)

    weights = dp_linear.adassp(X, y, epsilon, delta, num_classes=1)
    weights = weights.detach().numpy()
    assert weights.shape == (5, 1), f"weights.shape: {weights.shape}"

    # Test is such that the first feature is the most predictive
    assert weights[0] > np.sum(weights[1:]), f"weights[0]: {weights[0]}, sum(weights[1:]): {np.sum(weights[1:])}"

    # First feature is exactly the first weight :)
    assert np.abs(weights[0] - 1.0) < 0.1, f"weights[0]: {weights[0]}"


def test_adassp_eps1():
    y = torch.randn(100, 1)

    X = torch.hstack([y, torch.ones(100, 1), 0.1 * torch.randn(100, 3)])
    epsilon = torch.tensor(1.0)
    delta = torch.tensor(0.01)

    weights = dp_linear.adassp(X, y, epsilon, delta, num_classes=1)
    weights = weights.detach().numpy()
    assert weights.shape == (5, 1), f"weights.shape: {weights.shape}"

    print(weights)
    # Test is such that the first feature is the most predictive
    assert weights[0] > np.mean(weights[1:]), f"weights: {weights}, sum(weights[1:]): {np.sum(weights[1:])}"



