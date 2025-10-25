"""Test the synchro module."""
from nose2.tools import params

import Conv_relus_MNIST as cvxnn
import torch


def test_cvx_loss():
  sign_patterns = torch.randint(0, 2, (10000, 100))
  data = torch.rand(10000, 784)

  layer = cvxnn.CustomCVXLayer_orig(d=784, num_neurons=100, num_classes=10)
  y_pred = layer(data, sign_patterns)
  assert y_pred.shape == (10000, 10)

  layer = cvxnn.CustomCVXLayer(d=784, num_neurons=100, num_classes=10)
  y_pred = layer(data[0], sign_patterns[0])
  assert y_pred.shape == (1, 10), f"y_pred.shape: {y_pred.shape}"
