'''
    File        : SingleDDC.py
    Author      : Philip J. Erickson
    Date        : September 18, 2013
    Description : Program for estimating Single Agent DDC models using:
                    - Rust NFP algorithm (Rust 87)
                    - Hotz and Miller CCP method (Hotz and Miller 93)
'''

import pandas as pd
import numpy as np
from scipy.sparse import diags


def u_flow(y, j, params):
    ''' Period return function '''
    return -.01 * params[1] * (1-j) * y - j*params[0]


def val_inner(y, params, beta, EV, replace):
    ''' Utility given next period milage '''
    if replace == 0:
        val = (np.exp(u_flow(y, 0, params) + beta*EV[0, :]) + 
               np.exp(u_flow(y, 1, params) + beta*EV[1, :]))
    else:
        val = ((np.exp(u_flow(y, 0, params) + beta*EV[0, 0]) + 
               np.exp(u_flow(y, 1, params) + beta*EV[1, 0])) * 
               np.ones(y.shape))
    return np.log(val)


'''
    Rust Values:
    params = [.9999, 11.7257, 2.4569, 0.0937, 0.4475, 0.4459, 0.0127]
    stateMax = 400000
    stateInt = 2500
    stateNum = 4
'''
def val_iter(params, stateMax, stateInt, stateNum):
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
    P = params[-stateNum:]  # Transition matrix
    P = diags(P, list(xrange(stateNum)), shape=(K, K)).todense()
    P = P.T
    r = np.array(list(xrange(K)))
    guess = np.zeros((2, K))
    EV = guess
    EVTemp = np.zeros((2, K))
    
    tol = 1e-6; maxIter = 1000; dif = 1; iterNum = 0  # Iteration bounds
    while dif > tol and iterNum < maxIter:
        EV1 = val_inner(r, flowParams, beta, EV, 0)
        EV2 = val_inner(r, flowParams, beta, EV, 1)
        EVTemp = np.vstack((EV1, EV2))
        EVTemp = np.dot(EVTemp, P)  # E[] over future mileage draws
        # Correct for end of value function
        EVTemp[:, -stateNum:] = np.tile(EVTemp[:, -(stateNum + 1)], 
                                        (1, stateNum))
        dif = np.amax(abs(EVTemp - EV))
        EV = EVTemp
        iterNum += 1
    return EV


def x_set(x, p):
    ''' Assign next period miles according to G(.|.) '''
    lb = 0  # Lower bound for probability interval
    count = 0
    for pr in p:
        if x >= lb and x < lb + pr:   
            return count
        lb = lb + pr
        count += 1
    return (count - 1)  # Control for precision error on complimentary prob.

def replace(params, stateMax, stateInt, stateNum, n, t, x, eps):
    ''' Generate replacement decisions '''
    obsNum = 0
    action = []
    xOut = []
    EV = valIter(params, stateMax, stateInt, stateNum)
    for i in range(0, n):
        xCum = 0
        for j in range(0, t):
            u0 = (u_flow(x[obsNum], 0, params[1:-stateNum]) + 
                  eps[obsNum][0] + params[0]*EV[0, xCum])
            u1 = (u_flow(x[obsNum], 1, params[1:-stateNum]) + 
                  eps[obsNum][1] + params[0]*EV[1, 0])
            action.append((u1 >= u0)[0] * 1)
            if action[obsNum] == 1:
                xCum = 0
            else:
                xCum += x[obsNum][0]
            xOut.append(xCum)
            obsNum += 1
    return np.array([xOut, action]).T

def rust_sim(params, stateMax, stateInt, stateNum, n, t):
    ''' Simulate Rust data '''
    p = params[-stateNum:]
    obs = n * t  # Number of observations to be simulated
    
    eps = np.random.uniform(0, 1, (obs, 2))
    eps = 0.577 - np.log(-np.log(eps))  # Quantile fn. for Type 1 EV dist.
    
    x = np.random.uniform(0, 1, (t, n))
    vec_set = np.vectorize(x_set)
    pArray = np.ndarray((1,), dtype=object)  # Prep for vectorized use
    pArray[0] = p
    x = vec_set(x, pArray)    
    x = x.reshape((obs, 1), order='F')  # "Fortran order", ie. column major
    
    unit = np.array(list(range(n)))
    unit = np.tile(unit, (t, 1))
    unit = unit.reshape((obs, 1), order='F')
    
    time = np.array(list(range(t)))
    time = np.tile(time, (n, 1))
    time = time.reshape((obs, 1))
    
    data = replace(params, stateMax, stateInt, stateNum, n, t, x, eps)
    data = hstack((unit, time, data))
    cols = ['ident', 'time', 'x', 'i']
    data = pd.DataFrame(data, columns=cols)
    data.x = data.x * stateInt
    return data
    
    