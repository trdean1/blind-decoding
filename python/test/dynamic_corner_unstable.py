import numpy as np
import sys

sys.path.insert(0,'..')
import dynamic
from FeasibleRegion import FeasibleRegion

trials = 100

Y =  [[-0.85105277,-0.63492999, 3.41237287, 1.20811145, 0.35892148, 2.41836825],
      [ 3.96976924,-0.73519243,-1.04010774, 0.12971196, 1.03251896,-2.80775547],
      [ 3.12355480, 2.95773796, 0.67356399, 0.87056107, 2.94202708, 0.68919778],
      [-2.31709338, 0.66587824, 1.21174544, 2.38172050,-1.41046808, 3.28820386]]

U_i = [[-0.06779237, -0.08776801,  0.03347982, -0.04695821],
       [-0.01830087, -0.05418955, -0.09233134,  0.06187484],
       [ 0.06205623, -0.02276697, -0.05832655, -0.08862112],
       [ 0.08272422, -0.06683513,  0.05076455,  0.04168607]]

biggest = 0

for i in range(trials):

    U_i = np.matrix(U_i)
    Y = np.matrix(Y)

    fs = FeasibleRegion( Y )

    U = dynamic.dynamic_solve( U_i, Y )
    
    UY = U*Y
    max_l_inf = np.max(abs(U*Y))
    if max_l_inf > biggest:
        biggest = max_l_inf

    if max_l_inf > 1 + 1e-6:
        print "Found infeasible BFS"
        print "UY = "
        print U*Y
        print max_l_inf
        exit()

print "All %d trials were feasible to within %e" % (trials,abs(biggest-1.0))
