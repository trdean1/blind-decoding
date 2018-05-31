import numpy as np
import sys

sys.path.insert(0,'..')
import dynamic

dims = [(2,2),(2,4),(3,3),(3,6),(4,4),(4,10),
        (5,5),(5,10),(6,6),(6,12),(8,8),(8,20)]

trials_per = 50

results = [0 for i in dims]

print "Testing if dynamic solver completes for various values"
print "Only reporting exceptions not accuracy..."

for ii, dim in enumerate(dims):
    n = dim[0]
    k = dim[1]
    print "\nn=%d, k=%d" % (n, k)

    for jj in range(trials_per):
        if jj != 0 and jj % 100 == 0:
            sys.stdout.write('#')
            sys.stdout.flush()

        try:
            UY = dynamic.trial(n,k)
            results[ii] += 1
        except KeyboardInterrupt:
            raise
        except:
            pass

print "Completed runs:"
for ii, dim in enumerate(dims):
    print "(%d, %d): %d out of %d" % (dim[0], dim[1], results[ii], trials_per)
