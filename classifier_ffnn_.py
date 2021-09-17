from typing import List, Any
from random import random
import math
import numpy as np
from sklearn.decomposition import PCA
from math import sqrt

d = 50

def CalcDistant(vec1, vec2):
    vec = np.multiply(np.array(vec1 - vec2), np.array(vec1 - vec2))
    return vec.sum()

#text is a tokenized list of words 
def vectorization(text, embed): 
    vec = np.zeros((1, d))
    num = 0;
    unknown = 0
    for i in text:
        if i in embed:
            vec += embed[i]
    return vec/np.linalg.norm(vec)

# preprocessing: lowering and splitting
def preprocessing(text):
    text = text.lower()
    stripped_text = ''
    
    for c in text:
        if ((not c.isalnum()) and (c != ' ') and (len(c.encode(encoding='utf_8')) == 1)): 
            stripped_text = stripped_text + ' ' + c + ' '
        else:
            stripped_text += c
    return stripped_text
    
#tokenization: getting stuff together
def tokenization(text):
    q = text.split(' ')
    while ('' in q):
        q.remove('')
    while ('>' in q):
        q.remove('>')
    while ('<' in q):
        q.remove('<')    
    return q
   
def init_params(layer_sizes,activation):
    d = {}
    for i in range(len(layer_sizes) - 1):
        if ((activation == "sigmoid")|(activation == "tanh")):
            #Xavier
            w = np.random.randn((layer_sizes[i]+1)*layer_sizes[i+1]) * sqrt(1.0/(layer_sizes[i]+1)*layer_sizes[i+1])
            w = np.reshape(w, (layer_sizes[i]+1, layer_sizes[i+1]))
            d[i] = w
        else:
            #He
            w = np.random.randn((layer_sizes[i]+1)*layer_sizes[i+1]) * sqrt(2.0/(layer_sizes[i]+1)*layer_sizes[i+1])
            w = np.reshape(w, (layer_sizes[i]+1, layer_sizes[i+1]))
            d[i] = w
    
    #last layer
    if ((activation == "sigmoid")|(activation == "tanh")):
            #Xavier
        w = np.random.randn((layer_sizes[-1]+1)*2) * sqrt(1.0/(2*(layer_sizes[-1]+1)))
        w = np.reshape(w, (layer_sizes[-1]+1, 2))
        d[len(layer_sizes) - 1] = w
    else:
            #He
        w = np.random.randn((layer_sizes[-1]+1)*2) * sqrt(2.0/(2*(layer_sizes[-1]+1)))
        w = np.reshape(w, (layer_sizes[-1]+1, 2))
        d[len(layer_sizes) - 1] = w
    return d

def tanh(weight, x): #X is a matrix, weight is a matrix - return matrix
    b = np.dot(x, weight)
    b = 1 + np.exp(-2*b.T)
    return np.array(2/b) - 1

def relu(weight, x): #X is a matrix, weight is a matrix  - return matrix
    b = np.dot(x, weight)
    return np.maximum(b.T, 0)

def sigmoid(weight, x): #X is a matrix, weight is a matrix - return matrix
    b = np.dot(x, weight)
    b = 1 + np.exp(-b.T)
    return np.array(1/b).T

def fully_connected(a_prev, W, activation):
    a_prev = np.insert(a_prev, 0, np.ones(a_prev.shape[0]), 1)
    if (activation == "sigmoid"):
        layer_out = sigmoid (W, a_prev)
    elif (activation == "tanh"):
        layer_out = tanh (W, a_prev)
    elif (activation == "relu"):
        layer_out = relu (W, a_prev)
    else:
        #linear
        layer_out = np.dot(a_prev, W)
        layer_out = layer_out

    return layer_out.T, a_prev, W

def ffnn(X, params, activation):
    cash = []
    for i in params:
        X, tmpcash1, tmpcash2 = fully_connected(X, params[i], activation)
        cash.append (tmpcash1)
        cash.append (tmpcash2)
    X = np.exp(X - np.amax(X, axis = 0))
    X = (X.T/X.sum(axis = 1)).T #softmax
    cash.append (X)
    return X, cash

def softmax_crossentropy(ZL, Y):
    Y = np.array(Y)
    log_likelihood = -np.log(np.dot(ZL.T, Y))
    loss = np.sum(log_likelihood) / len(Y)
    math_expect = np.sum(np.dot(ZL.T, Y))
    return loss

def fully_connected_backward(dA, cache, activation):
    if (activation == "sigmoid"):
        derivative = cache[2]*(1-cache[2])
    elif (activation == "tanh"):
        derivative = 1 - cache[2]*cache[2]
    elif (activation == "relu"):
        derivative = np.maximum(cache[2], 0)
    else:
        #linear
        derivative = cache[2]
    if (dA.shape[1] == derivative.shape[0]):
        tmp = np.dot(dA,derivative)
    else:
        tmp = np.dot(dA.T,derivative)
    check_gradients(activation, cache[2], derivative)
    DZ = np.dot (cache[3], dA.T)*derivative.T
    DW = np.dot(cache[0].T, DZ.T)
    return np.delete(DZ.T, (0), axis = 1), DW

def ffnn_backward(dZL, caches, activation):
    d_param = {}
    dA = dZL
    d_param[-int(len(caches)/2)] = dA
    d_param[int(len(caches)/2)] = np.dot(caches[len(caches) - 4].T,dA)
    for i in range (int(len(caches)) - 4, 0, -2):
        d_param[- int(i/2) - 1], d_param[int(i/2) + 1] = fully_connected_backward(dA, caches[i-2:i+3], activation)
        dA = d_param[- int(i/2) - 1]
    #positive/2 - DZ 
    #negative/2 - DW
    return d_param

def softmax_crossentropy_backward(cache):
    dZL = cache[len(cache) - 2] - cache[len(cache) - 1] # ^Y - Y
    return np.array(dZL)
    
def check_gradients (activation, x, grad):
    h = 0.00001
    if (activation == "sigmoid"):
        correct = (1/(1 + np.exp(-(x + h).T)) - 1/(1 + np.exp(-(x - h).T)))/(2*h)
    elif (activation == "tanh"):
        correct = (2/(1 + np.exp(-2*(x + h).T)) - 2/(1 + np.exp(-2*(x - h).T)))/(2*h)
    elif (activation == "relu"):
        correct = (np.maximum(x + h, 0) - np.maximum(x - h, 0))/(2*h)
    else:
        #linear
        correct = 1
    print ("error = ", np.sum(abs(correct - grad.T).sum(axis = 0)/correct.shape[1]))
    return
    
def gd_step(params,grads,learning_rate, alpha):
    alpha = 0.0001
    for i in range (len(params)):
        if (params[i].shape[1] != grads[i+2].shape[1]):
            params[i] = params[i] - np.delete(learning_rate*grads[i+2], (0), axis=1) + alpha*params[i]
        else:
            params[i] = params[i] - learning_rate*grads[i+2] + alpha*params[i]
    return params
    
def train_ffnn(Xtrain, Ytrain, Xdev, Ydev, layer_sizes, learning_rate, num_epochs, batch_size):
    Xtrain -= Xtrain.mean(axis = 0)
    Xtrain /= Xtrain.std(axis = 0)
    X = np.zeros((Ytrain.shape[0], Ytrain.shape[1]))
    print(Ytrain)
    dict_weights = init_params(layer_sizes,"tanh")
    cur_batch = 0
    alpha = 0.001
    for i in range (num_epochs):
        if ((i == 500)|(i == 700)|i == 900):
            learning_rate /= 2
            alpha /= 2
        if (cur_batch + batch_size > Xtrain.shape[0]):
            cur_batch = 0
            
        #X[cur_batch:cur_batch + batch_size], cache = ffnn(np.insert(Xtrain[cur_batch:cur_batch + batch_size], 0, np.ones(batch_size), 1), dict_weights,"tanh")
        X[cur_batch:cur_batch + batch_size], cache = ffnn(Xtrain[cur_batch:cur_batch + batch_size], dict_weights,"tanh")
        cache.append(np.array(Ytrain[cur_batch:cur_batch + batch_size]))

        print (i, "] precision = ", np.sum(abs((X[cur_batch:cur_batch + batch_size]-Ytrain[cur_batch:cur_batch + batch_size]))/(2*batch_size)))
        
        dict_grads = ffnn_backward(softmax_crossentropy_backward(cache),cache,"tanh")
        #for i in (dict_grads.keys()):
        
        dict_weights = gd_step(dict_weights, dict_grads, learning_rate, alpha)
        cur_batch += batch_size

    return dict_weights
    

def train(
        train_texts: List[str],
        train_labels: List[str],
        pretrain_params: Any = None) -> Any:
    
    embeddings = {}
    f = open('glove.6B.50d.txt','r')
    for line in f:
        k = line.index(' ')
        embeddings[line[:k]] = np.fromstring(line[k:], sep = ' ')
    matr = np.ones((1, d))
    
    doc_amount = len(train_texts) # amount of documents in train sample
    y_labels = []
    for i in range(doc_amount):
        new_text = tokenization(preprocessing(train_texts[i]))
        matr = np.append(matr, vectorization(new_text, embeddings), axis=0)
        
        if (train_labels[i] == 'neg'):
            y_labels = y_labels + [[0, 1]]
        else:
            y_labels = y_labels + [[1, 0]]
            
    matr = np.delete(matr, (0), axis=0)
    learning_rate = 0.01
    num_epochs = 1200
    batch_size = 5000
    layer_sizes = [d, 200, 100]
    dict_weights = train_ffnn(matr, np.array(y_labels), layer_sizes, learning_rate, num_epochs, batch_size)
    
    return {'weights': dict_weights, 'embeddings':embeddings}  



def classify(texts: List[str], params: Any) -> List[str]:
    
    embeddings = params['embeddings']
    dict_weights = params['weights']

    doc_amount = len(texts) # amount of documents in train sample
    matr = np.ones((1, d))
    for i in range(doc_amount):
        new_text = tokenization(preprocessing(texts[i]))
        matr = np.append(matr, vectorization(new_text, embeddings), axis=0)
    
    matr = np.delete(matr, (0), axis=0)
    print("matr shape ", matr.shape)
    X, cache = ffnn(matr, dict_weights, "tanh")
    
    texts_labels = []
    print (X.shape)
    for i in range(X.shape[0]):
        if (X[i][0] > X[i][1]):
            texts_labels.append('pos')
        else:
            texts_labels.append('neg')
    return texts_labels


