import sys

sys.path.insert(0,'..')
import solver
import Xmats

reps_per_k = 400
k_range = [i for i in range(6, 15, 1)]
n = 5
trapped = [{} for i in range(len(k_range))]

def try_increment( d, k ):
    try:
        d[k] += 1
    except KeyError:
        d[k] = 1

for ii,k in enumerate(k_range):
    if ii != 0:
        print ''
    print "(%d, %d)" % (n, k)
    for jj in range(reps_per_k):
        if jj != 0 and jj % 10 == 0:
            sys.stdout.write('#')
            sys.stdout.flush()

        X = Xmats.X_guarantee(n,k)
        ret = solver.single_run(n,k)
        try_increment( trapped[ii], ret ) 

print "\n\nRESULTS"
for ii, k in enumerate(k_range):
    print "k = %2d:\n%s" % (k, trapped[ii])

