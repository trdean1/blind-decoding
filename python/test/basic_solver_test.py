import sys

sys.path.insert(0,'..')
import solver

dims = [(4,5),(4,6),(4,10),(4,14),(4,20),(4,30)]
trials_per = 300

results = [{} for i in range(len(dims))]

print "Testing with X to guarantee recover up to an ATM"
print "Dimensions:"
print dims
print "Trials per dimension: %d" % trials_per

for ii, dim in enumerate(dims):
    n = dim[0]
    k = dim[1]
    print "\nn=%d, k=%d" % (n, k)

    for jj in range(trials_per):
        if jj != 0 and jj % 100 == 0:
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
