import sys

sys.path.insert(0,'..')
import solver

dims = [(2,3),(2,4),(3,4),(3,6),(4,5),(4,10),
        (5,6),(5,10),(6,12),(8,20)]
trials_per = 100

results = [{} for i in range(len(dims))]

print "Testing with X to guarantee recover up to an ATM"
print "Dimensions:"
print dims
print "Trials per dimension: %d" % trials_per

for ii, dim in enumerate(dims):
    n = dim[0]
    k = dim[1]
    print "\nn=%d, k=%d" % (n, k)
    if n == 8:
        trials_per = 20

    for jj in range(trials_per):
        if jj != 0 and jj % 10 == 0:
            sys.stdout.write('#')
            sys.stdout.flush()

        res = solver.single_run(n,k)

        try:
            results[ii][res] += 1
        except KeyError:
            results[ii][res] = 1

print "\n\nResults:"
print "-"*60
for ii,res in enumerate(results):
    print "n=%d, k=%d" % (dims[ii][0], dims[ii][1])
    print res
