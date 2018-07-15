import sys

sys.path.insert(0,'..')
import solver

dims = [(5,10)]
trials_per = 10

for ii, dim in enumerate(dims):
    n = dim[0]
    k = dim[1]
    print "\nn=%d, k=%d" % (n, k)

    for jj in range(trials_per):
        res = solver.single_run(n,k)

        print res  
