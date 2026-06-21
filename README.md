# Overview

This repository contains the code for the paper "Convex Approximation of Two-Layer ReLU Networks for Hidden State Differential Privacy".

Abstract:
```
The hidden state threat model of differential privacy (DP) assumes that the adversary has access only to the final trained machine learning (ML) model, without seeing intermediate states during training. The current privacy analyses under this model, however, are limited to convex optimization problems, reducing their applicability to multi-layer neural networks, which are essential in modern deep learning applications.
Additionally, the most successful applications of the hidden state privacy analyses in classification tasks have only been for logistic regression models.
We demonstrate that it is possible to privately train convex problems with privacy-utility trade-offs comparable to those of 2-layer ReLU networks trained with DP stochastic gradient descent (DP-SGD).
We achieve this through a stochastic approximation of a dual formulation of the ReLU minimization problem, which results in a strongly convex problem. This enables the use of existing hidden state privacy analyses and provides accurate privacy bounds also for the noisy cyclic mini-batch gradient descent (NoisyCGD) method with fixed disjoint mini-batches.
Our experiments on benchmark classification tasks show that NoisyCGD can achieve privacy-utility trade-offs comparable to DP-SGD applied to 2-layer ReLU networks. Additionally, we provide theoretical utility bounds highlighting the speed-ups gained through the convex approximation.
```

[Arxiv link to the paper](https://arxiv.org/abs/2407.04884). Note that an amendment has been made to Lemma 3.1 in June 2026.

# Code

Running the iterative algorithms can be done with the scripts in the `scripts` folder. Those provide an example of how to run the algorithms. For example:

```
# Basic command to run the script for MNIST dataset
# Arguments are in order:
# beta, lr, sigma, batch_size, P, seed, feature_type

python Conv_relus_MNIST.py 0.01 0.00031 5.0 1024 100 42 0
```

The folder `notebooks` contains a notebook to run the SSP experiments on all three datasets.

The folder `dp_accounting` contains the code for the DP accounting. The Jupyter notebook `plot_epsilons.ipynb` shows how to plot the epsilon values for the different datasets and methods.

# LICENSE

This project is licensed under the MIT License. See the LICENSE.txt file for details.

The code is based on an earlier repository: https://github.com/tolgaergen/convex_nn
That repository does not have a license, but the authors have given permission for us to use it.

The accounting, in particular the gdp.py file, is based on the repository (it includes mu-GDP privacy analysis for NoisyCGD):
https://github.com/jinhobok/shifted_interpolation_dp


# Citing this work

If you use this code, please cite the following work:

```
@article{ergen2024convex,
  title={Convex Approximation of Two-Layer ReLU Networks for Hidden State Differential Privacy},
  journal={Advances in neural information processing systems (NeurIPS)},
  author={Romijnders, Rob and Koskela, Antti},
  year={2025},
  url={https://arxiv.org/abs/2407.04884}
}
```

# Contact

For questions about the code and paper, please contact romijndersrob@gmail.com and/or antti.h.koskela@nokia-bell-labs.com.
