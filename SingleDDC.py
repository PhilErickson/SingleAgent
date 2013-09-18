'''
    File        : SingleDDC.py
    Author      : Philip J. Erickson
    Date        : September 18, 2013
    Description : Code for estimating Single Agent DDC models using:
                    - Rust NFP algorithm (Rust 87)
                    - Hotz and Miller CCP method (Hotz and Miller 93)
'''

import pandas as pd
import numpy as np
from scipy.sparse import diags

def uFlow(y, j, params):
    # Period return function
    return -.01 * params[1] * (1 - j) * y - j * params[0]

def valInner(y, params, beta, EV, replace):
    # Utility given next period's milage
    if replace == 0:
        val = (np.exp(uFlow(y, 0, params) + beta * EV[0, :]) + 
               np.exp(uFlow(y, 1, params) + beta * EV[1, :]))
    else:
        val = ((np.exp(uFlow(y, 0, params) + beta * EV[0, 0]) + 
               np.exp(uFlow(y, 1, params) + beta * EV[1, 0])) * 
               np.ones(y.shape))
    return np.log(val)

'''
    Rust Values:
    params = [.9999, 11.7257, 2.4569, 0.0937, 0.4475, 0.4459, 0.0127]
    stateMax = 400000
    stateInt = 2500
    stateNum = 4
'''
def valIter(params, stateMax, stateInt, stateNum):
    '''
        FN      : Rust value fn iteration proceedure
        Inputs  : params:
                    - Discount factor
                    - Utl. flow params
                    - Transition probabilities
                  stateMax: maximum value of state space
                  stateInt: interval sizes for state space
                  stateNum: cardinality of movement options on state space
    '''
    beta = params[0]
    flowParams = params[1:-stateNum]
    K = stateMax / stateInt
    P = params[-stateNum:] # Transition matrix
    P = diags(P, list(xrange(stateNum)), shape = (K, K)).todense()
    P = P.T
    #r = np.linspace(0, stateMax, (stateMax / stateInt)) # state variable
    r = np.array(list(xrange(K)))
    guess = np.zeros((2, K))
    EV = guess
    EVTemp = np.zeros((2, K))
    tol = 1e-6; maxIter = 1000; dif = 1; iterNum = 0 # Iteration bounds
    while dif > tol and iterNum < maxIter:
        EV1 = valInner(r, params, beta, EV, 0)
        EV2 = valInner(r, params, beta, EV, 1)
        EVTemp = np.vstack((EV1, EV2))
        EVTemp = np.dot(EVTemp, P)
        #EVTemp[1, :] = EVTemp[1, 0]
        # Correct for end of value function
        EVTemp[:, -stateNum:] = np.tile(EVTemp[:, -(stateNum + 1)], 
                                        (1, stateNum))
        dif = np.amax(abs(EVTemp - EV))
        EV = EVTemp
        iterNum += 1
    return EV

#def rustSim(params, stateMax, stateInt, stateNum):
    