import json
import math
import random
import time
from dataclasses import dataclass, fields, field
from typing import List
from enum import Enum

import numpy as np
from numpy.f2py.auxfuncs import throw_error
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
    activations:list[Matrix | Vector] = field(default_factory=list)
    zs:list[Matrix | Vector] = field(default_factory=list)
    biases:list[Vector] = field(default_factory=list)

    def getInputLayerSize(self, n:Network)->int:
        return n.layer_sizes[0]
    def getOutputLayerSize(self, n:Network)->int:
        return n.layer_sizes[-1]

@dataclass
class TrainConfig(DictAccessMixin):
    dataset_count:int
    epoch_amount:int
    data_size:int
    batch_size:int
    learning_rate:float
    lwCa:dict


class Values:
    def __init__(self, x:Matrix|Vector = np.zeros(shape=(0,0)) , y:Matrix|Vector = np.zeros(shape=(0,0))):
        # if type(x) != type(y):
        #     raise TypeError(f"Input Output Type Shape Mismatch!")
        self.initialized:bool = not(x.shape == (0,0) or y.shape == (0,0))
        self.x = x
        self.y = y
        self.shape = self.x.shape
    #
    # def safe_x(self)->Matrix|Vector:
    #     if not self.initialized or self.x :
    #         raise ValueError(f"values Are Yet to be Initalized!")
    #     return self.x

    @staticmethod
    def get_batch_data(values:Values, batch:int, batchSize:int)->Values:
        if not values.initialized:
            raise ValueError(f"Values Are Yet to be Initaliized!")
        if batchSize * batch >= values.shape[0]:
            raise ValueError(f"Invalid Batch Error! {batchSize * batchSize} > {values.shape[0]}")

        return Values(values.x[(batch-1) * batchSize : batch * batchSize], values.y[(batch-1) * batchSize : batch * batchSize])


class Data:
    def __init__(self, layer_sizes:list[int], train_config:TrainConfig, train_datasets: list[Values] | None = None, test_datasets : Values| None = None, batchAxis:int = 1, neuronAxis:int = 0):
        if 1 < batchAxis < 0 or 1 < neuronAxis < 0:
            raise ValueError("Invalid Batch or Neuron Axis! Must be a positive Index between (0-1)!")
        if train_datasets is not None:
            raise TypeError("train_datasets can't be None!")
        self.layer_sizes = layer_sizes
        self.train_config = train_config
        self.train_datasets:list[Values] = train_datasets if train_datasets is not None else []
        self.test_datasets:Values|None = test_datasets
        self.initialized = train_datasets != []
        self.batchAxis = batchAxis
        self.neuronAxis = neuronAxis


    def  safe_test_datasets(self)->Values:
        if self.test_datasets is None:
            raise ValueError("Test Datasets can't be None!")
        return self.test_datasets
    def check_init(self):
        if not self.initialized:
            raise ValueError(f"Dataset Are Yet to be Initaliized!")

    def get_batch_data(self, dataset_id:int, batch:int)->Values:
        self.check_init()
        epoch = self.get_epoch(dataset_id)
        return Values.get_batch_data(epoch, batch, self.train_config.batch_size)

    def get_epoch(self, dataset_id:int)->Values:
        self.check_init()
        indices = np.random.permutation([i for i in range(self.train_datasets[dataset_id].shape[self.batchAxis])])
        return Values(np.array([self.train_datasets[dataset_id].x[i] for i in indices]), np.array([self.train_datasets[dataset_id].y[i] for i in indices]))

    def get_batch_size(self, dataset_id)->int:
        self.check_init()
        return (self.train_datasets[dataset_id]).shape[self.batchAxis] / self.train_config.batch_size

    def add_train_dataset(self, values:Values):
        if values is None:
            raise TypeError("Values cant be None!")

        if values.shape[self.batchAxis] % self.train_config.batch_size != 0:
            raise ValueError(f"Epoch Size indivisible by Batch Size!")
        self.train_datasets.append(values)

    def add_test_dataset(self, values:Values):
        if values.shape[self.batchAxis] % self.train_config.batch_size != 0:
            raise ValueError(f"Epoch Size indivisible by Batch Size!")

        self.test_datasets = values


class DropOut:
    def __init__(self, on:bool = False, rate = 0):
        self.on = on
        self.rate = rate

    def get_activations(self, activations:list[Matrix|Vector])->list[Matrix|Vector]:
        if not self.on:
            return activations
        return  [
            rng.binomial(1, 1 - self.rate, size=(activations[i].shape[0], 1)) * activations[i] / (1-self.rate)
            for i in range(len(activations))
        ]

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

class CType(Enum):
    BY_VALUE = "BY_VALUE"
    BY_GLOBAL_NORM = "BY_GLOBAL_NORM"
    NONE = "NONE"

class Clipping:
    def __init__(self, c_type:CType, c_const:float):
        if c_const < 0.0:
            raise ValueError("c_const can't be negative!")
        self.c_type = c_type
        self.c_const = c_const

    def get_wb(self, dw:list[Matrix], db:list[Vector])->tuple[list[Matrix], list[Vector]]:
        if self.c_type == CType.NONE:
            return dw, db
        match self.c_type:
            case CType.BY_GLOBAL_NORM:
                g = 0
                for matrix in dw:
                    g += np.sum(np.square(matrix))
                for matrix in db:
                    g += np.sum(np.square(matrix))
                g = np.sqrt(g)
                if g <= self.c_const:
                    return dw, db
                scale = min(1, self.c_const / g)
                return [weights * scale for weights in dw], [bias * scale for bias in db]
            case CType.BY_VALUE:
                updated_w = []
                updated_b = []
                for matrix in dw:
                    updated_w.append(np.maximum(-self.c_const, np.minimum(matrix, self.c_const)))
                for matrix in db:
                    updated_b.append(np.maximum(-self.c_const, np.minimum(matrix, self.c_const)))
                return updated_w, updated_b


def relu(z: np.ndarray):
    return np.maximum(0, z)
def relu_derivative(x):
  return (x > 0).astype(float)

#x input matrix shape (a, m)
#w weight matrix shape (b, a)
#m is the batch size.
def forward(x:Matrix|Vector, network:Network):
    Ws:list[Matrix] = network.weights #tensor: multiple weights matrix for each l+1 layers
    Bs:list[Vector] = network.biases

    curX:Matrix|Vector = x
    for i in range(1, len(Ws)):
        network.zs[i] = Ws[i] @ curX + Bs[i]
        network.activations[i] = relu(network.zs[i])
        curX = network.activations[i]


def sum_weights(weights:list[Matrix])->float:
    return sum(np.sum(np.abs(w)) for w in weights)

#rs = regulization constant (hyperparam)
def backwards(eval:Matrix|Vector, network:Network, step:int,
              reg_config:Regularization = Regularization(RType.NONE),
              scheduler_config:Scheduler = NoScheduler(),
              clipping:Clipping = Clipping(CType.NONE)):
    if eval.shape != network.activations[-1].shape:
        print(f"Shape Mismatch Error! Network LL:{network.activations[-1].shape} != Eval: {eval.shape}")
        return
    if len(network.activations) == 0 or network.size == 0:
        print(f"Error! Invalid Network Size!")
        return
    if  eval.shape[0] == 0 or eval.shape[1] == 0:
        print(f"Error! Eval Shape Empty.")
        return
    Ws:list[Matrix] = network.weights
    As:list[Matrix|Vector] = network.activations
    Zs:list[Matrix|Vector] = network.zs

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
    dWs, dBs = clipping.get_wb(dWs, dBs)
    for i in range(lastIndex+1):
        network.weights[lastIndex - i] -= lr * dWs[i]
        network.biases[lastIndex - i] -= lr * dBs[i]


def validate(x:Matrix|Vector, y:Matrix|Vector, network:Network):
    forward(x, network)
    return 1/(2 * y.shape[1]) * np.sum((network.activations[-1] - y))

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

def load_train_data(nn:Network) -> Data | None:
    with open("train.json", "r") as f:
        trainDataFile = json.load(f)
        shape_ = trainDataFile["shape"]
        if shape_["in"] != nn.layer_sizes[0] or shape_["out"] != nn.layer_sizes[-1]:
            print("Invalid Train Data Shape")
            return None
        iter_det = trainDataFile["iteration_details"]
        learn_det = trainDataFile["learn_details"]
        if iter_det["data_size"] / iter_det["epoch__count"] % learn_det["batch_size"] != 0:
            print("Batch amount doesn't divide Data per Epoch!")
            return None

        values:list[dict] = trainDataFile["values"]["train"]


        trainConfig: TrainConfig = TrainConfig(
            iter_det["dataset_count"],
            iter_det["epoch_count"],
            iter_det["data_size"],
            learn_det["batch_size"],
            learn_det["learning_rate"],
            learn_det["lw_ca"])
        data:Data = Data([shape_["in"]] + shape_["hidden"]  + [shape_["out"]], trainConfig)


        for n, epoch in enumerate(values):
            train_value = Values()
            for i, input, eval in enumerate(epoch.items()):
                train_value.x[n, i] = np.array(input)
                train_value.y[n, i] = np.array(eval)
            data.add_train_dataset(train_value)

        values: list[dict] = trainDataFile["values"]["test"]
        for n, epoch in enumerate(values):
            test_value = Values()
            for i, input, eval in enumerate(epoch.items()):
                test_value.x[n, i] = np.array(input)
                test_value.y[n, i] = np.array(eval)
            data.add_test_dataset(test_value)

        return data


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
    if train_data is None:
        return
    # batch_size = train_data.train_config.batch_size

    for i_dataset in range(train_data.train_config.dataset_count):
        eval_losses:list[float] = []
        for i_epoch in range(train_data.train_config.epoch_amount):
            batch_count = train_data.get_batch_size(i_dataset)
            for i_batch in range(batch_count):
                batch_dat = train_data.get_batch_data(i_dataset, i_batch)
                if do_config.on:
                    network.activations = do_config.get_activations(activations_original)
                forward(batch_dat.x, network)
                backwards(batch_dat.y, network, i_epoch,
                          scheduler_config=CosineDecay(train_data.train_config.lwCa["lr_start"],
                                                       train_data.train_config.lwCa["lr_max"],
                                                       train_data.train_config.lwCa["lr_end"],
                                                       train_data.train_config.lwCa["warmup_epoch"],
                                                       train_data.train_config.epoch_amount),
                          reg_config=Regularization(RType.L2))
            eval_loss = validate(train_data.safe_test_datasets().x, train_data.safe_test_datasets().y, network)
            eval_losses.append(eval_loss)
            if len(eval_losses) > 1:
                if eval_losses[i_epoch-1] < eval_losses[i_epoch]:
                    print(f"Early Stop to Prevent Overfitting!")
                    break
