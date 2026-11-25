import json
import math
import random
import time
from dataclasses import dataclass, fields, field
from typing import List
from enum import Enum

import numpy as np
from numpy.random._examples.cffi.extending import rng

# Define reusable rank-specific aliases (Python 3.12+)
type Vector[T: np.generic] = np.ndarray[tuple[int], np.dtype[T]]
type Matrix[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]
type Tensor3[T: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[T]]

class DictAccessMixin:
    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        setattr(self, key, value)

    def keys(self):
        return [f.name for f in fields(self)]

@dataclass
class Network(DictAccessMixin):
    size: int = -1
    layer_sizes: list[int] = field(default_factory=list)
    weights:list[Matrix] = field(default_factory=list)
    activations:list[Matrix] = field(default_factory=list)
    zs:list[Vector] = field(default_factory=list)
    biases:list[Vector] = field(default_factory=list)

@dataclass
class TrainDat(DictAccessMixin):
    dataset_count:int
    epoch_amount:int
    data_size:int
    batch_size:int
    learning_rate:float
    inputs:Tensor3
    evals:Tensor3
    lwCa:dict


class SType(Enum):
    STEP = "STEP"
    EXPONENTIAL = "EXPONENTIAL"
    COSINE = "COSINE"
    NONE = "NONE"

class Scheduler:
    def __init__(self, type:SType):
        self.type:SType = type

    def get_lr(self, step):
        raise NotImplementedError

class DropOut:
    def __init__(self, on:bool = False, rate = 0):
        self.on = on
        self.rate = rate

    def get_activations(self, activations:list[Matrix])->list[Matrix]:
        if not self.on:
            return activations
        return  [
            rng.binomial(1, 1 - self.rate, size=(activations[i].shape[0], 1)) * activations[i] / (1-self.rate)
            for i in range(len(activations))
        ]

class StepDecay(Scheduler):
    def __init__(self, initial_lr, factor, interval):
        super().__init__(SType.STEP)
        self.initial_lr = initial_lr
        self.factor = factor
        self.interval = interval

    def get_lr(self, step):
        return self.initial_lr * self.factor ** (step / self.interval)


class ExponentialDecay(Scheduler):
    def __init__(self, initial_lr, decay_rate):
        super().__init__(SType.EXPONENTIAL)
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def get_lr(self, step):
        return self.initial_lr * self.decay_rate ** step

class NoScheduler(Scheduler):
    def __init__(self):
        super().__init__(SType.NONE)

    def get_lr(self, step):
        return 0.001

class CosineDecay(Scheduler):
    def __init__(self, start, max_, final, warmup_rounds, epoch_amount):
        super().__init__(SType.COSINE)
        self.start = start
        self.max_ = max_
        self.final = final
        self.warmup_rounds = warmup_rounds
        self.epoch_amount = epoch_amount

    def get_lr(self, step):
        return lw_ca(self.start, self.max_, self.final, self.warmup_rounds, self.epoch_amount, step)

class RType(Enum):
    L1 = "L1"
    L2 = "L2"
    BOTH = "BOTH"
    NONE = "NONE"

class Regularization:

    def __init__(self, type:RType, lambda_value=None):
        if lambda_value is None:
            lambda_value = []
        self.type = type
        self.lambda_value = lambda_value

    def get_dw(self, weights:list[Matrix]):
        if self.type == RType.NONE: return 0.0

        match self.type:
            case RType.BOTH:
                return sum_weights(weights) * self.lambda_value[0] + self.lambda_value[0]/ 2 * sum_weights([np.square(weight) for weight in weights])
            case RType.L1:
                return sum_weights(weights) * self.lambda_value[0]
            case RType.L2:
                return self.lambda_value[0]/ 2 * sum_weights([np.square(weight) for weight in weights])



def relu(z: np.ndarray):
    return np.maximum(0, z)
def relu_derivative(x):
  return (x > 0).astype(float)

#x input matrix shape (a, m)
#w weight matrix shape (b, a)
#m is the batch size.
def forward(x:Matrix, nn:Network):
    Ws:list[Matrix] = nn.weights #tensor: multiple weights matrix for each l+1 layers
    Bs:list[Vector] = nn.biases

    curX:Matrix = x
    for i in range(1, len(Ws)):
        nn.zs[i] = Ws[i] @ curX + Bs[i]
        nn.activations[i] = relu(nn.zs[i])
        curX = nn.activations[i]


def sum_weights(weights:list[Matrix])->float:
    return sum(np.sum(np.abs(w)) for w in weights)

#rs = regulization constant (hyperparam)
def backwards(eval:np.ndarray, network:Network, step:int, reg_config:Regularization = Regularization(RType.NONE), scheduler_config:Scheduler = NoScheduler()):
    Ws:list[Matrix] = network.weights
    As:list[Matrix] = network.activations
    Zs:list[Vector] = network.zs

    latest_loss= np.zeros(Zs[-1].shape[0])
    dWs:list[Matrix] = []
    dBs:list[Vector] = []
    m = As[0].shape[1]
    lr = scheduler_config.get_lr(step)
    for i in range(len(Ws)-1, -1, -1):
        if network.size-i-1 == 0:
            loss = (As[i] - eval) / m
        else:
            loss = Ws[i+1].T @ latest_loss

        dZ = loss * relu_derivative(Zs[i])
        latest_loss = loss

        dWs.append(dZ @ As[i - 1].T + reg_config.get_dw(network.weights) * sum_weights(network.weights))
        dBs.append(np.sum(dZ, axis=1))

    lastIndex = len(dWs) - 1
    for i in range(lastIndex+1):
        network.weights[lastIndex - i] -= lr * dWs[i]
        network.biases[lastIndex - i] -= lr * dBs[i]


def load_network(networkIndex:int = 0)->Network:
    networkDetails:Network = Network()
    with open("net.json", "r") as f:
        nn = json.load(f)["networks"][networkIndex]
        traits:list[str] = list(json.load(f)["contents"])

        for trait in traits:
            match trait:
                case "size":
                    networkDetails.size = int(nn[trait])
                    break
                case "layer_shape":
                    networkDetails.layer_sizes = list(nn[trait])
                    break
                case _:
                    networkDetails[trait] = np.array(nn[trait])
                    break

    return networkDetails

def load_train_data(nn:Network):
    with open("train.json", "r") as f:
        trainDataFile = json.load(f)
        if trainDataFile["shape"]["in"] != nn.layer_sizes[0] or trainDataFile["shape"]["out"] != nn.layer_sizes[-1]:
            print("Invalid Train Data Shape")
            return None
        iter_det = trainDataFile["iteration_details"]
        learn_det = trainDataFile["learn_details"]
        if iter_det["data_size"] / iter_det["epoch__count"] % learn_det["batch_size"] != 0:
            print("Batch amount doesn't divide Data per Epoch!")
            return None

        values:list[dict] = trainDataFile["values"]

        inputs:Tensor3
        evals:Tensor3

        for n, epoch in enumerate(values):
            for i, input, eval in enumerate(epoch.items()):
                inputs[n][i] = np.array(input)
                evals[n][i] = np.array(eval)


        trainData:TrainDat = TrainDat(
            iter_det["dataset_count"],
            iter_det["epoch_count"],
            iter_det["data_size"],
            learn_det["batch_size"],
            learn_det["learning_rate"],
            inputs, evals,
            learn_det["lw_ca"])

        return trainData

def getRandom(min:float, max:float):
    r = (round(time.time() * 1000) + random.randint(0, 1)) % 2
    return min + r * (max - min)

def get_normal_randoms(amt:int, min:float, max:float):
    rands = []
    for i in range(amt):
        rands.append(getRandom(min, max))
    return rands

def he_init(nn:Network):
    for i in range(1, nn.size):
        layer_size = nn.layer_sizes[i-1]
        a = math.sqrt(6/layer_size)
        nn.weights[i] = np.array(get_normal_randoms(layer_size, -a, a))


def handle_init_weights(nn:Network):
    if (nn.weights[0].shape[0] == 0  and nn.weights[1].shape[0] == 0 or
            nn.weights[0].size == 0 and nn.weights[1].size == 0):
        he_init(nn)

def stepDecay(init_lr:float, decay_factor:float, s:int, t:int)->float:
    return init_lr * decay_factor ** (t/s)

def expDecay(init_lr:float, decay_rate:float, t)->float:
    return init_lr * math.exp(-decay_rate * t)

def consineAnnealing(lr_max:float, lr_min:float, epoch_count:int, t:int):
    return lr_min + 0.5 * (lr_max-lr_min) * (1 + math.cos(math.pi * t / epoch_count))

def linear_warmup(lr_start:float, lr_end:float, warmup_epoch_count:int, t:int)->float:
    return lr_start + t/warmup_epoch_count * (lr_end-lr_start)

def lw_ca(lr_start, lr_max, lr_end, warmupAmount:int, totalEpoch:int, t:int)->float:
    return linear_warmup(lr_start, lr_max, warmupAmount, t) \
        if t > warmupAmount \
        else consineAnnealing(lr_max, lr_end, totalEpoch, t)


def run(do_config: DropOut = DropOut()):
    network = load_network()
    activations_original = network.activations.copy()
    handle_init_weights(network)
    train_data = load_train_data(network)
    if not train_data:
        return
    batch_size = train_data.batch_size
    for i_dataset in range(train_data.dataset_count):
        for i_epoch in range(train_data.epoch_amount):
            indices = [i for i in range(train_data.inputs.shape[1])]
            np.random.permutation(indices)
            in_epoch = np.array([train_data.inputs[i_dataset][indices[i]] for i in range(train_data.inputs.shape[1])])
            ev_epoch = np.array([train_data.evals[i_dataset][indices[i]] for i in range(train_data.evals.shape[1])])

            for i_batch in range(int(train_data.inputs.shape[1]/batch_size)):
                in_ = in_epoch[i_batch * batch_size : min((i_batch+1) * batch_size, train_data.inputs.shape[1])]
                eval = ev_epoch[i_batch * batch_size : min((i_batch+1) * batch_size, train_data.inputs.shape[1])]
                if do_config.on:
                    network.activations = do_config.get_activations(activations_original)
                forward(in_, network)
                backwards(eval, network, i_epoch,
                          scheduler_config= CosineDecay(train_data.lwCa["lr_start"],
                                               train_data.lwCa["lr_max"],
                                               train_data.lwCa["lr_end"],
                                               train_data.lwCa["warmup_epoch"],
                                               train_data.epoch_amount),
                          reg_config= Regularization(RType.L2))





