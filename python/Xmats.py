import numpy as np

#XXX: This is not all MPS just a couple each
MSP = { 2 : [
        np.matrix([
            [1, 1],
            [1, -1]
        ])
    ],
    3 : [
        np.matrix([
            [ 1,  1,  1],
            [ 1,  1, -1],
            [ 1, -1,  1],       
        ]),
        np.matrix([
            [-1,  1,  1],
            [ 1, -1,  1],
            [ 1,  1, -1],       
        ])
    ],
    4 : [
        np.matrix([
            [ 1,  1,  1,  1],
            [ 1,  1, -1, -1],
            [ 1, -1,  1, -1],
            [ 1, -1, -1,  1]
        ]),
        np.matrix([
            [ 1,  1,  1, -1],
            [ 1,  1, -1,  1],
            [ 1, -1,  1,  1],
            [-1,  1,  1,  1]
        ])
    ],
    5 : [
        np.matrix([
            [ -1,  -1,  -1,  -1,  -1],
            [ -1,  -1,  -1,   1,   1],
            [ -1,  -1,   1,  -1,   1],
            [ -1,   1,  -1,  -1,   1],
            [ -1,   1,   1,   1,  -1],
        ]),
        np.matrix([
            [ -1,  -1,  -1,  -1,  -1],
            [ -1,  -1,  -1,   1,   1],
            [ -1,  -1,   1,  -1,   1],
            [ -1,   1,   1,   1,  -1],
            [ -1,   1,  -1,  -1,   1],
        ]),
    ],
    6 : [
        np.matrix([
            [-1, -1, -1,  1,  1,  1], 
            [-1, -1,  1, -1,  1, -1], 
            [ 1, -1, -1, -1, -1, -1], 
            [-1,  1, -1, -1,  1,  1], 
            [ 1,  1, -1,  1,  1, -1], 
            [ 1, -1,  1, -1,  1,  1],
        ]),
        np.matrix([
            [ 1, -1,  1,  1, -1, -1], 
            [-1, -1, -1, -1,  1, -1], 
            [ 1, -1, -1, -1, -1,  1], 
            [-1,  1, -1,  1, -1, -1], 
            [-1, -1,  1,  1, -1,  1], 
            [ 1, -1, -1,  1,  1,  1],
        ])
    ],
    8 : [
        np.matrix([
            [  1,  1,  1,  1,  1,  1,  1,  1],
            [  1,  1, -1, -1,  1,  1, -1, -1],
            [  1, -1, -1,  1,  1, -1, -1,  1],
            [  1, -1,  1, -1,  1, -1,  1, -1],
            [  1,  1,  1,  1, -1, -1, -1, -1],
            [  1,  1, -1, -1, -1, -1,  1,  1],
            [  1, -1, -1,  1, -1,  1,  1, -1],
            [  1, -1,  1, -1, -1,  1, -1,  1],
        ]),
        np.matrix([
            [  1,  1,  1, -1,  1,  1,  1, -1],
            [  1,  1, -1,  1,  1,  1, -1,  1],
            [  1, -1,  1,  1,  1, -1,  1,  1],
            [ -1,  1,  1,  1, -1,  1,  1,  1],
            [  1,  1,  1, -1, -1, -1, -1,  1],
            [  1,  1, -1,  1, -1, -1,  1, -1],
            [  1, -1,  1,  1, -1,  1, -1, -1],
            [ -1,  1,  1,  1,  1, -1, -1, -1],
        ])
    ]
}

###XXX: Again not all, just a few...
guarantee = {
    2 : [
        np.matrix([
            [1, 1],
            [1, -1]
        ])
    ],
    3 : [
        np.matrix([
            [ 1,  1,  1],
            [ 1,  1, -1],
            [ 1, -1,  1],       
        ]),
        np.matrix([
            [-1,  1,  1],
            [ 1, -1,  1],
            [ 1,  1, -1],       
        ])
    ],
    4 : [
        np.matrix([
            [ 1,  1,  1,  1,  1],
            [ 1,  1, -1, -1,  1],
            [ 1, -1,  1, -1,  1],
            [ 1, -1, -1,  1, -1]
        ]),
        np.matrix([
            [ 1,  1,  1, -1,  1],
            [ 1,  1, -1,  1,  1],
            [ 1, -1,  1,  1,  1],
            [-1,  1,  1,  1,  1]
        ])
    ],
    5 : [
        np.matrix([
            [ -1,  -1,  -1,  -1,  -1, -1],
            [ -1,  -1,  -1,   1,   1, -1],
            [ -1,  -1,   1,  -1,   1,  1],
            [ -1,   1,  -1,  -1,   1,  1],
            [ -1,   1,   1,   1,  -1,  1],
        ]),
        np.matrix([
            [ -1,  -1,  -1,  -1,  -1, -1],
            [ -1,  -1,  -1,   1,   1, -1],
            [ -1,  -1,   1,  -1,   1,  1],
            [ -1,   1,   1,   1,  -1,  1],
            [ -1,   1,  -1,  -1,   1,  1],
        ]),
    ],
}

def X_MSP(n, k, strict=False):
    X = []
    try:
        X = MSP[n][np.random.randint(len(MSP[n]))]
        if k > n:
            X = np.concatenate( (X, X_random(n, k-n)), axis=1 )

        return X
    except KeyError:
        if strict:
            raise
        #If we don't have an MSP return a random matrix
        return X_random(n,k)

def X_guarantee(n, k, strict=False):
    X = []
    try:
        X = guarantee[n][np.random.randint(len(guarantee[n]))]
        kx = X.shape[1]
        if k > kx:
            X = np.concatenate( (X, X_random(n, k-kx)), axis=1 )

        return X
    except KeyError:
        if strict:
            raise
        #If we don't have a guarantee just return an MSP
        return X_MSP(n, k)

def X_random(n, k):
    X = np.array( np.random.rand(n,k) < 0.5, dtype = np.int) * 2 - 1
    return np.matrix(X)
