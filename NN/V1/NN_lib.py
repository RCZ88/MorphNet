import json


class Neuron:
    def __init__(self, id,  activation:float, z:float, weights:list[float], bias:float):
        self.id = id
        self.w:list[float] = weights
        self.b:float = bias
        self.a = activation
        self.z = z

    def calculate_activation(self):
        sum = 0
        for weight in self.w:
            sum += weight
        sum += self.b
        return sum

class Layer:
    def __init__(self, neuronsId:list[int]):
        self.neuronsId:list[int] = neuronsId
        self.size = len(self.neuronsId)

    def __init__(self):
        self.neuronsId = []
        self.size = 0

    def addNeuron(self, neuron:Neuron):
        self.neuronsId.append(neuron.id)
        self.size += 1

