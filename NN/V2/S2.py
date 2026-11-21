import json
import math
import random
import time
from dataclasses import dataclass, fields, field
from typing import TypedDict, List

import numpy as np



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
    weights:Tensor3 = field(default_factory=lambda: np.empty((0,0,0), dtype=np.float32))
    activations:Matrix = field(default_factory=lambda: np.empty((0,0), dtype=np.float32))
    zs:Matrix = field(default_factory=lambda: np.empty((0,0), dtype=np.float32))
    biases:Matrix = field(default_factory=lambda: np.empty((0,0), dtype=np.float32))

@dataclass
class TrainDat(DictAccessMixin):
    epoch_amount:int
    data_size:int
    batch_size:int
    learning_rate:float
    inputs:Tensor3
    evals:Tensor3




def relu(z: np.ndarray):
    return np.maximum(0, z)
def relu_derivative(x):
  return (x > 0).astype(float)

def forward(x:np.ndarray, nn:Network):
    Ws:Tensor3 = nn.weights #tensor: multiple weights matrix for each l+1 layers
    Bs:Matrix = nn.biases
    As:Matrix = nn.activations
    As[0] = x
    for i in range(1, Ws.shape[0]):
        As[i] = relu(As[i-1] @ Ws[i].T + Bs[i])

def backwards(eval:np.ndarray, nn:Network, m:int, lr:float = 0.0001):
    Ws:Tensor3 = nn.weights
    As:Matrix = nn.activations
    Zs:Matrix = nn.zs

    latest_dZ= np.zeros(Zs.shape)
    dWs:list[Matrix] = []
    dBs:list[Vector] = []
    for i in range(eval.shape[0]-1, 0, -1):
        if nn.size-i-1 == 0:
            dA = (As[i] - eval) / m
        else:
            dA = Ws[i+1].T @ latest_dZ

        dZ = dA * relu_derivative(Zs[i])
        latest_dZ = dZ

        dWs.append(dZ @ As[i - 1].T)
        dBs.append(np.sum(dZ, axis=1))

    lastIndex = len(dWs) - 1
    for i in range(lastIndex+1):
        nn.weights[lastIndex-i] -= lr * dWs[i]
        nn.biases[lastIndex-i] -= lr * dBs[i]


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


        trainData:TrainDat = TrainDat(iter_det["epoch_count"],
                                      iter_det["data_size"],
                                      learn_det["batch_size"],
                                      learn_det["learning_rate"],
                                      inputs, evals)
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



def run():
    load_network()

    load_train_data()