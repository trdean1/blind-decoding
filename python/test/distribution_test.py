import numpy as np
import sys

sys.path.insert(0,'..')
import dynamic

n = 4
tol = 1e-6

for k in range(5, 20, 2):
    good = 0.0
    zero = 0.0
    other = 0.0
    total = 0.0

    print( "n=%d, k=%d" % (n,k) )
    for i in range(2000):
        if i % 100 == 0 and i != 0:
            sys.stdout.write('#')
            sys.stdout.flush()

        try:
            UY = dynamic.trial(n,k)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print "Warning skipping exception:"
            print e
            continue

        gg = np.asscalar( sum(sum( abs((abs(UY) - 1.0)) < tol ).transpose() ) )
        good += gg

        zz = np.asscalar( sum(sum( abs(UY) < tol ).transpose() ) )
        zero += zz

        other += n*k - (gg + zz)
        total += n*k


    sys.stdout.write('\n')
    print("pm 1: %f,\t0: %e\tother: %e\n" % (good/total, zero/total, other/total) )
