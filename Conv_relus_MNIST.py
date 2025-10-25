import sys
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from opacus.data_loader import DPDataLoader

from torch.func import functional_call,  grad
import torch.nn.functional as F
import torch.nn as nn

FEATURE_TYPE = {
    0: "ConvexReLU",
    1: "RFF",
}

if __name__ == "__main__":
    # Example command line call:
    # python Conv_relus_MNIST.py 0.01 0.00031 0.1 32 100 42
    #                           beta  lr    sigma bs  P  seed

    #L2-Regularization constant
    beta = float(sys.argv[1])

    learning_rate=float(sys.argv[2])
    sigma=float(sys.argv[3])
    batch_size = int(sys.argv[4])

    # Number of randomly chosen hyperplanes
    P = int(sys.argv[5])

    seed_ = int(sys.argv[6])

    feature_type = int(sys.argv[7])
    assert feature_type in FEATURE_TYPE, f"Invalid feature type: {feature_type}"
    feature_type = FEATURE_TYPE[feature_type]
    print(sys.argv)
else:

    beta = 0.01
    learning_rate=0.00031
    sigma=0.1
    batch_size = 32
    P = 100
    seed_ = 42
    feature_type = FEATURE_TYPE[0]

verbose = True

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
print(f"TORCH: Using device: {device}")

# Clipping constant
C_G = 1.0

np.random.seed(seed_)
torch.manual_seed(seed_)
torch.cuda.manual_seed(seed_)


def to_numpy(x):
  """Convert a torch tensor or numpy array to a numpy array"""
  if isinstance(x, torch.Tensor):
    return x.numpy()
  elif isinstance(x, np.ndarray):
    return x
  else:
    raise ValueError(f"Unsupported type: {type(x)}")


class PrepareData3D(Dataset):
    def __init__(self, X, y, z):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X

        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y
        self.y=self.y.unsqueeze(-1)

        if not torch.is_tensor(z):
            self.z = torch.from_numpy(z)
        else:
            self.z = z

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.z[idx]

class PrepareData(Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X

        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


one = torch.ones(1)

def clip_vector_norm(X,R):
    X_clipped = X /torch.maximum(one.to(X.device), X.norm(2, dim=-1)/R)[:, None]
    return X_clipped

def generate_sign_patterns(A, P, verbose=False):
    # generate sign patterns
    n, d = A.shape
    sign_pattern_list = []
    u_vector_list = []

    umat = np.random.normal(0, 1, (d,P))
    umat1 = torch.randn((d,P))

    # print(umat.shape, umat1.size())
    sampled_sign_pattern_mat = np.matmul(A, umat) >= 0
    for i in range(P):
        sampled_sign_pattern = sampled_sign_pattern_mat[:,i]
        sign_pattern_list.append(sampled_sign_pattern)
        u_vector_list.append(umat[:,i])
    if verbose:
        print("Number of sign patterns generated: " + str(len(sign_pattern_list)))
    return len(sign_pattern_list),sign_pattern_list, np.array(u_vector_list)

def generate_rff(X, n_components, gamma=1.0):
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

    return Z, W, b

def generate_rff_test(X, W, b, n_components):
    Z = np.sqrt(2/n_components) * np.cos(X @ W + b)
    return Z

directory = './datasets/'

normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
transform_normalize = transforms.Lambda(lambda x: (x / 255.0) * 2 - 1)

train_dataset = datasets.MNIST(
    directory, train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
test_dataset =  datasets.MNIST(
    directory, train=False, transform=transforms.Compose([transforms.ToTensor()]))


A_train = train_dataset.data.to(torch.float)
A_test = test_dataset.data.to(torch.float)
y = train_dataset.targets
y_test = test_dataset.targets

A_train = A_train.view(A_train.shape[0], -1)
A_test = A_test.view(A_test.shape[0], -1)
num_samples_train, num_features = A_train.size()
print(f"A shape: {A_train.shape}")
print(f"A type: {type(A_train)}")


if feature_type == "ConvexReLU":
    num_neurons, sign_pattern_list, u_vector_list = generate_sign_patterns(A_train, P, verbose)
    sign_patterns = torch.stack(sign_pattern_list).int().transpose(0, 1).numpy()
    u_vectors = torch.Tensor(u_vector_list).reshape((num_neurons, A_train.shape[1])).T
    ds_train = PrepareData3D(X=A_train, y=y, z=sign_patterns)
    ds_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False)

    sign_patterns_test = torch.matmul(A_test, u_vectors) >= 0
    ds_test = PrepareData3D(X=A_test, y=y_test, z=sign_patterns_test)
    ds_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
elif feature_type == "RFF":
    A_train = transform_normalize(train_dataset.data.to(torch.float))
    A_test = transform_normalize(test_dataset.data.to(torch.float))
    A_train = A_train.view(A_train.shape[0], -1)
    A_test = A_test.view(A_test.shape[0], -1)

    gamma = 1.0 / (num_features * A_train.std()) # Heuristic for choosing gamma
    num_neurons = P
    print(f"gamma: {gamma}")

    Z_train, W, b = generate_rff(A_train, P, gamma=gamma)
    ds_train = PrepareData3D(X=A_train, y=y, z=Z_train)
    ds_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False)

    Z_test = generate_rff_test(A_test, W, b, P)
    ds_test = PrepareData3D(X=A_test, y=y_test, z=Z_test)
    ds_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
else:
    raise ValueError(f"Invalid feature type: {feature_type}")

class CustomLossFunction(nn.Module):
    def __init__(self,beta,d):
        self.beta= beta
        self.d = d
        super(CustomLossFunction, self).__init__()

    def forward(self,yhat, y, v):

        loss = 0.5 * torch.norm(yhat - y)**2 # +   self.beta/2 * torch.sum(torch.norm(v, dim=1)**2))

        return loss

class CustomCVXLayer(torch.nn.Module):
    def __init__(self, d, num_neurons, num_classes=10, do_extend=False):
        super(CustomCVXLayer, self).__init__()
        self.do_extend = do_extend
        if not do_extend:
            self.v = torch.nn.Parameter(data=torch.zeros(num_neurons, d, num_classes), requires_grad=True)
        else:
            self.v = torch.nn.Parameter(data=torch.zeros(num_neurons, num_classes), requires_grad=True)

    def forward(self, x, sign_patterns):
        if self.do_extend:
            return torch.matmul(x, self.v)

        if len(sign_patterns.size()) > 1:
            sign_patterns = sign_patterns.unsqueeze(2)
        else:
            sign_patterns = sign_patterns.unsqueeze(-2)

        x = x.unsqueeze(-2)
        Xv_w = torch.matmul(x, self.v) # P x N x C
        sp_Xv_w = sign_patterns.float() @ Xv_w.permute(1, 0, 2)
        y_pred = torch.sum(sp_Xv_w, dim=1, keepdim=False) # N x C

        return y_pred

def validation_cvxproblem(model, testloader, beta, device):
    torch_255 = torch.tensor(255.0).to(device)

    test_loss = []
    test_correct =[]

    loss_func = CustomLossFunction(beta, num_features)
    with torch.no_grad():
        v=model.state_dict()["v"]
        for ix, (_x, _y, _z) in enumerate(testloader):
            _x = _x.view(_x.shape[0], -1)

            _x = _x.to(device)
            _y = _y.to(device).squeeze()
            _z = _z.to(device)

            _x = _x / torch_255

            _y_oh = F.one_hot(_y, num_classes=10)

            yhat = model(_x, _z)

            loss = loss_func(yhat, _y_oh, v)

            test_correct.append(torch.eq(torch.argmax(yhat, dim=1), _y).cpu().to(torch.float).mean())
            test_loss.append(loss.item())


    return np.mean(test_loss), np.mean(test_correct)

class CustomCVXLayer_orig(torch.nn.Module):
    def __init__(self, d, num_neurons, num_classes=10, do_extend=False):
        super(CustomCVXLayer_orig, self).__init__()
        self.do_extend = do_extend
        if not do_extend:
            self.v = torch.nn.Parameter(data=torch.zeros(num_neurons, d, num_classes), requires_grad=True)
        else:
            self.v = torch.nn.Parameter(data=torch.zeros(num_neurons, num_classes), requires_grad=True)

    def forward(self, x,  sign_patterns):
        if self.do_extend:
            return torch.matmul(x, self.v)

        sign_patterns = sign_patterns.unsqueeze(2)

        x = x.view(x.shape[0], -1) # n x d
        Xv_w = torch.matmul(x, self.v) # P x N x C
        y_pred = torch.sum(torch.mul(sign_patterns, Xv_w.permute(1, 0, 2)), dim=1, keepdim=False) # N x C
        return y_pred


loss_func = nn.CrossEntropyLoss()

one = torch.ones(1, device=device)
def clip_vector_norm(X,R):
    X_clipped = X /torch.maximum(one, X.norm(2, dim=-1)/R)[:, None]
    return X_clipped

def compute_loss(params, batch, y):
    _x, _z = batch
    yhat = functional_call(model_, (params), (_x, _z))
    loss = loss_func(yhat, y)

    return loss


if __name__ == "__main__":
    num_classes = 10
    n_epochs = 400
    model_ = CustomCVXLayer(num_features, num_neurons, num_classes)
    model_test_ = CustomCVXLayer_orig(num_features, num_neurons, num_classes)

    model_.to(device)
    model_test_.to(device)

    model_weights = {k: v.detach() for k, v in model_.named_parameters()}

    clip_gradients= torch.vmap(clip_vector_norm,in_dims=(0,None))
    grad_ = grad(compute_loss)
    ft_compute_sample_grad = torch.vmap(grad_, in_dims=(None, 0, 0))

    test_accs = []
    optimizer = torch.optim.SGD(model_.parameters(), lr=learning_rate, momentum=0.0)
    for epoch in range(n_epochs):
        for ix, (_x, _y, _z) in enumerate(ds_train):
            optimizer.zero_grad()

            _x, _y, _z = _x.to(device), _y.to(device), _z.to(device)

            ft_per_sample_grad = ft_compute_sample_grad(model_weights,(_x, _z),_y)

            flattened_grads = ft_per_sample_grad['v'].view(ft_per_sample_grad['v'].shape[0], -1)
            noise_std = sigma * C_G / len(_y)
            for p in model_.parameters():
                p.grad = (clip_gradients(flattened_grads, C_G).sum(0)/(batch_size)).view(P, num_features, num_classes)
                p.grad.add_(noise_std * torch.randn_like(p.grad))
                p.grad.add_(beta * p.data) # Problem is beta-strongly convex

            optimizer.step()

        # Make test model parameters match the training model parameters
        with torch.no_grad():
            for ptrain, ptest in zip(model_.parameters(), model_test_.parameters()):
                ptest.copy_(ptrain)

        test_loss, test_correct = validation_cvxproblem(model_test_, ds_test, beta, device) # loss on the entire test set
        test_accs.append(test_correct)

        print(f"Epoch [{epoch+1:3}], TEST: cvx loss:  {test_loss:.5f} acc: {100*test_correct:.2f}%")

    np.save('./mnist_results/test_accs_hiddenstate_' + str(sigma) + '_' + str(beta) + '_' + str(learning_rate) + '_' + str(P) + '_' + str(seed_),test_accs)
