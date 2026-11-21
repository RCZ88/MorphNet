import math
from math import *

import numpy
from numpy.ma.core import argmax
from transformers import GPT2LMHeadModel
import numpy as np
import json

global qDim
global kDim
global vDim

# global vocab
# global embeddings

def softmax(sList:list):
    sm:list[float] = []

    sumOfList = sum([exp(x) for x in sList])
    for i in range(len(sList)):
       sm[i] = exp(sList[i])/sumOfList

    return sm

def loadJson():
    with open("vocab.json", "r") as file:
        vocab_data = json.load(file)
        vocab = vocab_data["tokens"]
    with open("embeddings.json", "r") as file:
        embeddings = json.load(file)["embeddings"]
    with open("params.json", "r") as file:
        params = json.load(file)
    return vocab, embeddings, params

type Tokenized = list[tuple[str, list[float]]]

def mapStringToToken(string:str, embeddings:list[list[float]] = [], vocab:dict[str, int] = {}) -> Tokenized:
    v, e, _ = loadJson()
    if len(embeddings) == 0: embeddings = e
    if len(vocab) == 0: vocab = v

    print(f"vocab length: {len(vocab)}")

    maxLenToken = 7
    i = 0
    j = i + maxLenToken
    rt:Tokenized= []
    while i < len(string)-1:
        sliced = string[i:j]
        print(sliced + ', ', end = "")
        if sliced in vocab:
            print()
            rt.append((sliced, embeddings[vocab[sliced]]))
            i = j
            j = i + maxLenToken
            continue
        j-=1
    return rt

def combineXs(rt:Tokenized)->np.ndarray:
    vectors = []
    for _, embedding in rt:
        vectors.append(embedding)

    return np.array(vectors)

def printResultTokens(rt:Tokenized):
    for token, embeddings in rt:
        concat = ""
        for embed in embeddings:
            concat += f" {embed}, "
        str = "{"  + concat[:-2] + "}"
        print(f'Token "{token}": {str}')


def getAttention(X):
    X_in_shape = X.shape
    X += generatePEs(X.shape[0])
    print(f"'X *= W_P' SHAPE: {X_in_shape} *= {W_P.shape} = {X.shape}")

    K = X @ W_K
    print(f"'K = X @ W_K' SHAPE: {X.shape} @ {W_K.shape} = {K.shape}")

    Q = X @ W_Q
    print(f"'Q = X @ W_Q' SHAPE: {X.shape} @ {W_Q.shape} = {Q.shape}")

    K_T = K.T
    print(f"'K_T = K.T' SHAPE: {K.shape}.T = {K_T.shape}")

    S = Q @ K_T
    print(f"'S = Q @ K_T' SHAPE: {Q.shape} @ {K_T.shape} = {S.shape}")

    d_k = W_K.shape[1]

    A = S.copy() / math.sqrt(d_k)
    print(f"'A = S.copy() / math.sqrt(d_k)' SHAPE: {S.shape} / scalar = {A.shape}")

    V = X @ W_V
    print(f"'V = X @ W_V' SHAPE: {X.shape} @ {W_V.shape} = {V.shape}")

    # print(f"W_V Shape = {W_V.shape}")
    O_head: np.ndarray = A @ V
    print(f"'O_head = A @ V' SHAPE: {A.shape} @ {W_V.shape} = {O_head.shape}")
    print("O_head dimension amt: ", O_head.ndim)
    O_concat = O_head#np.concatenate(O_head, axis=1)
    # oCount = O_head.shape[1]
    # for i in range(oCount):
    #     O_concat.extend(O_head[i])

    O_concat_arr = np.array(O_concat)
    print(f"'O_concat = np.array(O_concat)' SHAPE: {O_head.shape} -> {O_concat_arr.shape}")
    return O_concat_arr


def normal_cdf(x):
    t = np.linspace(-10, x, 10000)

    y = np.exp(-(t ** 2) / 2) / np.sqrt(2 * np.pi)

    return np.trapezoid(y, t)

def GELU(x):
    cdf_val = np.vectorize(normal_cdf)(x)
    gelu_out = x * cdf_val
    print(f"'GELU = x * normal_cdf(x)' SHAPE: {x.shape} + {cdf_val.shape} = {gelu_out.shape}")
    return gelu_out

def generatePEs(tokCount:int):
    print(f"dmodel: {d_model}")
    PE = np.zeros((tokCount, d_model))
    for i in range(tokCount):
        for j in range(d_model):
            if j % 2 == 0:
                PE[i, j] =math.sin(j/math.pow(10000, 2*j/d_model))
            else:
                PE[i, j] =math.cos(j/math.pow(10000, 2*j/d_model))

    return PE


def normalize(matrix, gamma,bias, epsi = 0.00001):
    X_squared = np.mean(np.square(matrix))
    print(f"'X_squared = np.mean(np.square(matrix))' SHAPE: {matrix.shape} -> {X_squared.shape}")

    norm =  np.zeros_like(matrix)
    for i in range(np.shape(matrix)[0]):
        variance = np.mean(np.square(matrix[i, :])) / matrix.shape[0]
        avg = np.mean(matrix[i, :])
        for j in range(np.shape(matrix)[1]):
            divider = math.sqrt(variance + epsi)
            # print(f"i: {i} j:{j}")
            norm[i, j] = (gamma[j] * (matrix[i][j] - avg))/divider + bias[j]

    print(f"'norm = gamma * matrix / divider' SHAPE: {gamma.shape} * {matrix.shape} = {norm.shape}")
    return norm

v, embeddings, params = loadJson()

epsi = 0.00001

d_model = params["architecture"]["d_model"]

p = params["parameters"]
W_K = np.array(p["W_K"])
W_Q = np.array(p["W_Q"])
W_P = np.array(p["P"])
W_V = np.array(p["W_V"])
W_O = np.array(p["W_O"])

normParam = params["normalization_parameters"]
gamma_1 = np.array(normParam["layer_norm_1"]["gamma"])
beta_1 = np.array(normParam["layer_norm_1"]["beta"])
gamma_2 = np.array(normParam["layer_norm_2"]["gamma"])
beta_2 = np.array(normParam["layer_norm_2"]["beta"])
gamma_fin = np.array(normParam["final_layer_norm"]["gamma"])

W_1 = np.array(p["W_1"])
W_2 = np.array(p["W_2"])
B_1 = np.array(p["b_1"])
B_2 = np.array(p["b_2"])

def recurseBlock(X, amount:int, count:int = 0):
    if (count == amount):
        res = gamma_fin * X / math.sqrt(np.mean(X**2) + epsi)
        print(f"'res = gamma_fin * X / math.sqrt(np.mean(X**2) + epsi)' SHAPE: {gamma_fin.shape} * {X.shape} = {res.shape}")
        return res

    X_norm_1 = normalize(X, gamma_1, beta_1)
    print(f"s{X_norm_1.shape}")
    attn_out = getAttention(X_norm_1)
    print(f"attn_out: ({attn_out.shape}), W_O: ({W_O.shape})")
    O_projection = attn_out @ W_O
    print(f"'O_projection = attn_out @ W_O' SHAPE: {attn_out.shape} @ {W_O.shape} = {O_projection.shape}")

    X_projection = X + O_projection
    print(f"'X_projection = X + O_projection' SHAPE: {X.shape} + {O_projection.shape} = {X_projection.shape}")

    X_norm_2 = normalize(X_projection, gamma_2, beta_2)
    Z_mul = X_norm_2 @ W_1
    print(f"'Z_mul = X_norm_2 @ W_1' SHAPE: {X_norm_2.shape} @ {W_1.shape} = {Z_mul.shape}")

    Z = Z_mul + B_1
    print(f"'Z = Z_mul + B_1' SHAPE: {Z_mul.shape} + {B_1.shape} = {Z.shape}")

    Z_gelu = GELU(Z)
    FFN_mul = Z_gelu @ W_2
    print(f"'FFN_mul = GELU(Z) @ W_2' SHAPE: {Z_gelu.shape} @ {W_2.shape} = {FFN_mul.shape}")

    FFN = FFN_mul + B_2
    print(f"'FFN = FFN_mul + B_2' SHAPE: {FFN_mul.shape} + {B_2.shape} = {FFN.shape}")

    X_l = X_projection + FFN
    print(f"'X_l = X_projection + FFN' SHAPE: {X_projection.shape} + {FFN.shape} = {X_l.shape}")

    return X_l

if __name__ == '__main__':
    userText = input("Input Something: ")

    rt = mapStringToToken(userText, embeddings, v)
    printResultTokens(rt)
    X = combineXs(rt)
    print(f"'X = combineXs(rt)' SHAPE: {X.shape}")

    X_final = recurseBlock(X, 5)
    print(f"'X_final = recurseBlock(X, 5)' SHAPE: {X_final.shape}")

    embeddings = np.array(embeddings)
    W_vocab = embeddings.T
    print(f"'W_vocab = e.T' SHAPE: {embeddings.shape}.T = {W_vocab.shape}")

    logits:numpy.ndarray = X @ W_vocab
    print(f"'logits = X @ W_vocab' SHAPE: {X.shape} @ {W_vocab.shape} = {logits.shape}")

    p_arr = []
    temp = 0.01
    sum_ez = 0
    for i in range(logits.shape[0]):
        exp_logits_2 = np.exp(logits[2, :] / temp)
        sum_ez += exp_logits_2
        print(f"'sum_ez += np.exp(logits[2, :] / temp)' SHAPE: {exp_logits_2.shape} = {sum_ez.shape}")

    output = ""

    for i in range(logits.shape[0]):
        exp_logits_i = np.exp(logits[i, :] / temp)
        p = exp_logits_i / sum_ez
        print(f"'p = np.exp(logits[i, :] / temp) / sum_ez' SHAPE: {exp_logits_i.shape} / {sum_ez.shape} = {p.shape}")
        aMax = argmax(p)
        print(f"ARGMAX = {aMax}")
        v_rev = {id: token for token, id in v.items()}
        if aMax in v_rev:
            output += v_rev[aMax]

    print(output)