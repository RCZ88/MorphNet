import json

import math
import random
import time

from NN_lib import *


def loadDatas():
    n_count = 0
    neurons = []
    layers = []
    with open("net.json", "r") as file:
        nn_dat = json.load(file)
        for item in nn_dat:
            layer = Layer()
            for neuron_data in item["neurons"]:
                neuron = Neuron(n_count, neuron_data["activation"], neuron_data["z"], neuron_data["weights"],
                                neuron_data["bias"])
                neurons.append(neuron)
                layer.addNeuron(neuron)
                n_count += 1
            layers.append(layer)

    with open("train.json", "r") as file:
        train_dat = json.load(file)
        inputs = list(map(lambda epoch: epoch["data"]["input"], train_dat['values']))
        evals = list(map(lambda epoch: epoch["data"]["eval"], train_dat['values']))

    return {"neurons":neurons, "layers":layers, "inputs":inputs, "evals":evals}


init_data = loadDatas()

layers:list[Layer] = init_data["layers"]
neurons:list[Neuron] = init_data["neurons"]
inputs:list[list[float]] = init_data["inputs"]
evals:list[list[float]] =  init_data["evals"]




def RELU(x:float):
    return max(0,x)
def TANH(x:float):
    return math.tanh(x)
def sigmoid(x:float):
    return 1 / (1 + math.exp(-x))
def sigmoid_deriv(x:float):
    return sigmoid(x) * (1-sigmoid(x))

def forward(input_dat:list[float], act_func):
    for fl_id in layers[0].neuronsId:
        n = neurons[fl_id]
        n.a = input_dat
    for i in range(1, len(layers)):
        prev_layer = layers[i-1]
        cur_layer = layers[i]

        for n, cl_id in enumerate(cur_layer.neuronsId):
            total = 0
            cl_neuron = neurons[cl_id]
            for j, pl_id in enumerate(prev_layer.neuronsId):
                pl_neuron = neurons[pl_id]
                total += pl_neuron.a * cl_neuron.w[j] + cl_neuron.b
            cl_neuron.z = total
            cl_neuron.a = act_func(total)

def back(ll_eval:list[float], lr:float = 0.001):
    for i in range(len(layers)-1, 0, -1):
        layer = layers[i]
        fl_losses:list[float] = []

        for n, cl_id in enumerate(layer.neuronsId):
            cl_n = neurons[cl_id]
            if i == len(layers) - 1:
                fl_losses.append((cl_n.a - ll_eval[n]) * sigmoid_deriv(cl_n.z))
            else:
                fl_ids:list[int] = layers[i+1].neuronsId
                sum = 0
                for j in range(0, len(fl_losses)):
                    fl_ns = neurons[fl_ids[j]]
                    sum += fl_ns.w[n] * fl_losses[j]
                fl_losses.append(sum * sigmoid_deriv(cl_n.z))

            for j, pl_id in enumerate(layers[i-1].neuronsId):
                pl_n = neurons[pl_id]
                cl_n.w[j] -= lr * (pl_n.a - fl_losses[-1])
            cl_n.b -= lr * fl_losses[-1]


def getRandom(min:float, max:float):
    r = (round(time.time() * 1000) + random.randint(0, 1)) % 2
    return min + r * (max - min)

def get_normal_randoms(amt:int, min:float, max:float):
    rands = []
    for i in range(amt):
        rands.append(getRandom(min, max))
    return rands

def he_init():
    for i in range(1, len(layers)):
        a = math.sqrt(6/layers[i-1].size)
        for n in layers[i].neuronsId:
            neurons[n].w = get_normal_randoms(len(layers[i-1].neuronsId), -a, a)

