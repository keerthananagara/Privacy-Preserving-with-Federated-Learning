"""
Useful tools
"""
import numpy as np
import random
import torch


def mnist_noniid(labels, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 30, 2000
    num_shards = int(num_users*3)
    num_imgs = int(60000 / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 3, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma * s, data_shape).to(device)

import numpy as np
from decimal import *
from scipy.special import comb

getcontext().prec = 128


def rdp2dp(rdp, bad_event, alpha):
    """
    convert RDP to DP, Ref:
    - Canonne, Clément L., Gautam Kamath, and Thomas Steinke. The discrete gaussian for differential privacy. In NeurIPS, 2020. (See Proposition 12)
    - Asoodeh, S., Liao, J., Calmon, F.P., Kosut, O. and Sankar, L., A better bound gives a hundred rounds: Enhanced privacy guarantees via f-divergences. In ISIT, 2020. (See Lemma 1)
    """
    return rdp + 1.0/(alpha-1) * (np.log(1.0/bad_event) + (alpha-1)*np.log(1-1.0/alpha) - np.log(alpha))


def compute_rdp(alpha, q, sigma):
    """
    RDP for subsampled Gaussian mechanism, Ref:
    - Mironov, Ilya, Kunal Talwar, and Li Zhang. R\'enyi differential privacy of the sampled gaussian mechanism. arXiv preprint 2019.
    """
    sum_ = Decimal(0.0)
    for k in range(0, alpha+1):
        sum_ += Decimal(comb(alpha, k)) * Decimal(1-q)**Decimal(alpha-k) * Decimal(q**k) * Decimal(np.e)**(Decimal(k**2-k)/Decimal(2*sigma**2))
    rdp = sum_.ln() / Decimal(alpha-1)
    return float(rdp)
 
    
def search_dp(q, sigma, bad_event, iters=1):
    """
    Given the sampling rate, variance of Gaussian noise, and privacy parameter delta, 
    this function returns the corresponding DP budget.
    """
    min_dp = 1e5
    for alpha in list(range(2, 101)):
        rdp = iters * compute_rdp(alpha, q, sigma)
        dp = rdp2dp(rdp, bad_event, alpha)
        min_dp = min(min_dp, dp)
    return min_dp
    
    
def calibrating_sampled_gaussian(q, eps, bad_event, iters=1, err=1e-3):
    """
    Calibrate noise to privacy budgets
    """
    sigma_max = 100
    sigma_min = 0.1
    
    def binary_search(left, right):
        mid = (left + right) / 2
        
        lbd = search_dp(q, mid, bad_event, iters)
        ubd = search_dp(q, left, bad_event, iters)
        
        if ubd > eps and lbd > eps:    # min noise & mid noise are too small
            left = mid
        elif ubd > eps and lbd < eps:  # mid noise is too large
            right = mid
        else:
            print("an error occurs in func: binary search!")
            return -1
        return left, right
        
    # check
    if search_dp(q, sigma_max, bad_event, iters) > eps:
        print("noise > 100")
        return -1
    
    while sigma_max-sigma_min > err:
        sigma_min, sigma_max = binary_search(sigma_min, sigma_max)
    return sigma_max

# Machine learning models
import torch
from torch import nn
from kymatio.torch import Scattering2D

class MNIST_CNN(nn.Module):
    """
    End-to-end CNN model for MNIST and Fashion-MNIST, with Tanh activations. 
    References:
    - Papernot, Nicolas, et al. Tempered Sigmoid Activations for Deep Learning with Differential Privacy. In AAAI 2021.
    - Tramer, Florian, and Dan Boneh. Differentially Private Learning Needs Better Features (or Much More Data). In ICLR 2021. 
    """
    def __init__(self, input_dim, output_dim):
        super(MNIST_CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        
        self.fc = nn.Sequential(nn.Linear(4 * 4 * 32, 32),
                                        nn.Tanh(),
                                        nn.Linear(32, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
def get_scatter_transform():
    shape = (28, 28, 1)
    scattering = Scattering2D(J=2, shape=shape[:2])
    K = 81 * shape[2]
    (h, w) = shape[:2]
    return scattering, K, (h//4, w//4)


class ScatterLinear(nn.Module):
    """
    ScatterNet model used in the following paper
    - Tramer, Florian, and Dan Boneh. Differentially Private Learning Needs Better Features (or Much More Data). In ICLR 2021. 
    See https://github.com/ftramer/Handcrafted-DP/blob/main/models.py
    """
    def __init__(self, in_channels, hw_dims, input_norm=None, classes=10, clip_norm=None, **kwargs):
        super(ScatterLinear, self).__init__()
        self.K = in_channels
        self.h = hw_dims[0]
        self.w = hw_dims[1]
        self.fc = None
        self.norm = None
        self.clip = None
        self.build(input_norm, classes=classes, clip_norm=clip_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None, bn_stats=None, clip_norm=None, classes=10):
        self.fc = nn.Linear(self.K * self.h * self.w, classes)

        if input_norm is None:
            self.norm = nn.Identity()
        elif input_norm == "GroupNorm":
            self.norm = nn.GroupNorm(num_groups, self.K, affine=False)
        else:
            self.norm = lambda x: standardize(x, bn_stats)

        if clip_norm is None:
            self.clip = nn.Identity()
        else:
            self.clip = ClipLayer(clip_norm)

    def forward(self, x):
        x = self.norm(x.view(-1, self.K, self.h, self.w))
        x = self.clip(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    
class LogisticRegression(nn.Module):
    """Logistic regression"""
    def __init__(self, num_feature, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_feature, output_size)

    def forward(self, x):
        return self.linear(x)

    
class MLP(nn.Module):
    """Neural Networks"""
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.Tanh(),

            nn.Linear(1000, output_dim))

    def forward(self, x):
        return self.model(x)
    
    
class three_layer_MLP(nn.Module):
    """Neural Networks"""
    def __init__(self, input_dim, output_dim):
        super(three_layer_MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 600),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(600, 300),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(300, 100),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(100, output_dim))

    def forward(self, x):
        return self.model(x)
    

class MnistCNN_(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MnistCNN_, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    
# Federated Learning Model in PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np
import copy


class FLClient(nn.Module):
    """ Client of Federated Learning framework.
        1. Receive global model from server
        2. Perform local training (compute gradients)
        3. Return local model (gradients) to server
    """
    def __init__(self, model, output_size, data, lr, E, batch_size, q, clip, sigma, device=None):
        """
        :param model: ML model's training process should be implemented
        :param data: (tuple) dataset, all data in client side is used as training data
        :param lr: learning rate
        :param E: epoch of local update
        """
        super(FLClient, self).__init__()
        self.device = device
        self.BATCH_SIZE = batch_size
        self.torch_dataset = TensorDataset(torch.tensor(data[0]),
                                           torch.tensor(data[1]))
        self.data_size = len(self.torch_dataset)
        self.data_loader = DataLoader(
            dataset=self.torch_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        self.sigma = sigma    # DP noise level
        self.lr = lr
        self.E = E
        self.clip = clip
        self.q = q
        if model == 'scatter':
            self.model = ScatterLinear(81, (7, 7), input_norm="GroupNorm", num_groups=27).to(self.device)
        else:
            self.model = model(data[0].shape[1], output_size).to(self.device)

    def recv(self, model_param):
        """receive global model from aggregator (server)"""
        self.model.load_state_dict(copy.deepcopy(model_param))

    def update(self):
        """local model update"""
        self.model.train()
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        # optimizer = torch.optim.Adam(self.model.parameters())
        
        for e in range(self.E):
            # randomly select q fraction samples from data
            # according to the privacy analysis of moments accountant
            # training "Lots" are sampled by poisson sampling
            idx = np.where(np.random.rand(len(self.torch_dataset[:][0])) < self.q)[0]

            sampled_dataset = TensorDataset(self.torch_dataset[idx][0], self.torch_dataset[idx][1])
            sample_data_loader = DataLoader(
                dataset=sampled_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=True
            )
            
            optimizer.zero_grad()

            clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
            for batch_x, batch_y in sample_data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x.float())
                loss = criterion(pred_y, batch_y.long())
                
                # bound l2 sensitivity (gradient clipping)
                # clip each of the gradient in the "Lot"
                for i in range(loss.size()[0]):
                    loss[i].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                    for name, param in self.model.named_parameters():
                        clipped_grads[name] += param.grad 
                    self.model.zero_grad()
                    
            # add Gaussian noise
            for name, param in self.model.named_parameters():
                clipped_grads[name] += gaussian_noise(clipped_grads[name].shape, self.clip, self.sigma, device=self.device)
                
            # scale back
            for name, param in self.model.named_parameters():
                clipped_grads[name] /= (self.data_size*self.q)
            
            for name, param in self.model.named_parameters():
                param.grad = clipped_grads[name]
            
            # update local model
            optimizer.step()



class FLServer(nn.Module):
    """ Server of Federated Learning
        1. Receive model (or gradients) from clients
        2. Aggregate local models (or gradients)
        3. Compute global model, broadcast global model to clients
    """
    def __init__(self, fl_param):
        super(FLServer, self).__init__()
        self.device = fl_param['device']
        self.client_num = fl_param['client_num']
        self.C = fl_param['C']      # (float) C in [0, 1]
        self.clip = fl_param['clip']
        self.T = fl_param['tot_T']  # total number of global iterations (communication rounds)

        self.data = []
        self.target = []
        for sample in fl_param['data'][self.client_num:]:
            self.data += [torch.tensor(sample[0]).to(self.device)]    # test set
            self.target += [torch.tensor(sample[1]).to(self.device)]  # target label

        self.input_size = int(self.data[0].shape[1])
        self.lr = fl_param['lr']
        
        # compute noise using moments accountant
        # self.sigma = compute_noise(1, fl_param['q'], fl_param['eps'], fl_param['E']*fl_param['tot_T'], fl_param['delta'], 1e-5)
        
        # calibration with subsampeld Gaussian mechanism under composition 
        self.sigma = calibrating_sampled_gaussian(fl_param['q'], fl_param['eps'], fl_param['delta'], iters=fl_param['E']*fl_param['tot_T'], err=1e-3)
        print("noise scale =", self.sigma)
        
        self.clients = [FLClient(fl_param['model'],
                                 fl_param['output_size'],
                                 fl_param['data'][i],
                                 fl_param['lr'],
                                 fl_param['E'],
                                 fl_param['batch_size'],
                                 fl_param['q'],
                                 fl_param['clip'],
                                 self.sigma,
                                 self.device)
                        for i in range(self.client_num)]
        
        if fl_param['model'] == 'scatter':
            self.global_model = ScatterLinear(81, (7, 7), input_norm="GroupNorm", num_groups=27).to(self.device)
        else:
            self.global_model = fl_param['model'](self.input_size, fl_param['output_size']).to(self.device)
        
        self.weight = np.array([client.data_size * 1.0 for client in self.clients])
        self.broadcast(self.global_model.state_dict())

    def aggregated(self, idxs_users):
        """FedAvg"""
        model_par = [self.clients[idx].model.state_dict() for idx in idxs_users]
        new_par = copy.deepcopy(model_par[0])
        for name in new_par:
            new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
        for idx, par in enumerate(model_par):
            w = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
            for name in new_par:
                # new_par[name] += par[name] * (self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users]))
                new_par[name] += par[name] * (w / self.C)
        self.global_model.load_state_dict(copy.deepcopy(new_par))
        return self.global_model.state_dict().copy()

    def broadcast(self, new_par):
        """Send aggregated model to all clients"""
        for client in self.clients:
            client.recv(new_par.copy())

    def test_acc(self):
        self.global_model.eval()
        correct = 0
        tot_sample = 0
        for i in range(len(self.data)):
            t_pred_y = self.global_model(self.data[i])
            _, predicted = torch.max(t_pred_y, 1)
            correct += (predicted == self.target[i]).sum().item()
            tot_sample += self.target[i].size(0)
        acc = correct / tot_sample
        return acc

    def global_update(self):
        # idxs_users = np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False)
        idxs_users = np.sort(np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False))
        for idx in idxs_users:
            self.clients[idx].update()
        self.broadcast(self.aggregated(idxs_users))
        acc = self.test_acc()
        torch.cuda.empty_cache()
        return acc

    def set_lr(self, lr):
        for c in self.clients:
            c.lr = lr

# Application of FL task

from torchvision import datasets, transforms
import torch
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


scattering, K, (h, w) = get_scatter_transform()
scattering.to(device)

def get_scattered_feature(dataset):
    scatters = []
    targets = []
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True, num_workers=1, pin_memory=True)

    
    for (data, target) in loader:
        data, target = data.to(device), target.to(device)
        if scattering is not None:
            data = scattering(data)
        scatters.append(data)
        targets.append(target)

    scatters = torch.cat(scatters, axis=0)
    targets = torch.cat(targets, axis=0)

    data = torch.utils.data.TensorDataset(scatters, targets)
    return data

def load_mnist(num_users):
    train = datasets.MNIST(root="~/data/", train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(root="~/data/", train=False, download=True, transform=transforms.ToTensor())
    
    # get scattered features
    train = get_scattered_feature(train)
    test = get_scattered_feature(test)
    
    train_data = train[:][0].squeeze().cpu().float()
    train_label = train[:][1].cpu()
    
    test_data = test[:][0].squeeze().cpu().float()
    test_label = test[:][1].cpu()

    # split MNIST (training set) into non-iid data sets
    non_iid = []
    user_dict = mnist_noniid(train_label, num_users)
    for i in range(num_users):
        idx = user_dict[i]
        d = train_data[idx]
        targets = train_label[idx].float()
        non_iid.append((d, targets))
    non_iid.append((test_data.float(), test_label.float()))
    return non_iid


client_num = 5
d = load_mnist(client_num)


"""
FL model parameters.
"""
import warnings
warnings.filterwarnings("ignore")

lr = 0.075

fl_param = {
    'output_size': 10,
    'K': K,
    'h': h,
    'w': w,
    'client_num': client_num,
    'model': 'scatter',
    'data': d,
    'lr': lr,
    'E': 500,
    'C': 1,
    'eps': 4.0,
    'delta': 1e-5,
    'q': 0.01,
    'clip': 0.1,
    'tot_T': 10,
    'batch_size': 128,
    'device': device
}

fl_entity = FLServer(fl_param).to(device)

import time

acc = []
start_time = time.time()
for t in range(fl_param['tot_T']):
    acc += [fl_entity.global_update()]
    print("global epochs = {:d}, acc = {:.4f}".format(t+1, acc[-1]), " Time taken: %.2fs" % (time.time() - start_time))