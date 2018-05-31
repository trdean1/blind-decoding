import numpy as np
import sys
import pickle

sys.path.insert(0,'..')
import dynamic

Y = []
Ui = []
with open('error.p', 'r') as fp:
    unpickler = pickle.Unpickler(fp)
    instance = unpickler.load()
    Y = instance['Y']
    Ui = instance['U']

[U,S,V] = np.linalg.svd(Y)
print "Singular values of Y:\n%s" % S

U = dynamic.dynamic_solve( Ui, Y, True )

print U*Y
