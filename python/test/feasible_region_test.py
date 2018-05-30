import numpy as np
import sys

sys.path.insert(0,'..')
import dynamic
from FeasibleRegion import FeasibleRegion


Y = [[-2.9559865737125,  2.5865278325864, -2.2980562996102,  0.7369790120865,
-1.7022470264114,  2.5865278325864,  2.5865278325864, -0.7369790120865,
-1.9907185593875 ],
[  1.2508223406517, -0.7159569731051,  2.9493672391056, -3.8027781425851,
3.6435053746187, -0.7159569731051, -0.7159569731051,  3.8027781425851,
1.4100951086181 ],
[ -0.9635641883542, -1.4738581147596, -2.9837682836541, -1.3922099258609,
-2.0144903304535, -1.4738581147596, -1.4738581147596,  1.3922099258609,
2.4431360679602 ],
[ -0.8759863225138, -0.8875888657254,  2.6402078223407, -1.6158556098730,
1.2462441219873, -0.8875888657254, -0.8875888657254,  1.6158556098730,
-0.5063748346281 ]]


U_i = [[ -0.0528368520860,  0.0018089247339, -0.0934020602292,  0.0640784675058],
[  0.1123168564818, -0.0154594953961, -0.0433106680742,  0.0299184522178 ],
[  0.0143772089628,  0.1204140139453, -0.0208770819374, -0.0219751804950 ],
[ -0.0033891996287, -0.0297186311760, -0.0677450265753, -0.1007021730710 ]]

U_i = np.matrix(U_i)
Y = np.matrix(Y)

fs = FeasibleRegion( Y )

###################################
# Basic test
###################################
print 'Insert first entry'
fs.insert( 0, 0 )

print 'Make full rank'
fs.insert( 0, 1 )
fs.insert( 0, 2 )
fs.insert( 0, 3 )

print 'Adding redundant constraint'
fs.insert( 0, 5 )

p = dynamic.orthogonalize_p( np.matrix( np.copy( Y.T ) ) )
for i in range(p.shape[0]):
    p[i,:] /= np.linalg.norm(p[i,:])

print 'Difference from old method: %e' % np.linalg.norm(fs.p[0] - p)
print '-'*60

print 'Removing row 5'
print 'Change in p: %f\n' % np.linalg.norm(fs.p[0] - p)
 
print 'Removing row 3'
fs.remove( 0, 3 )
print 'Difference with old method: %f\n' % np.linalg.norm(fs.p[0] - p[0:3,:])

print 'Removing row 1'
fs.remove( 0, 1 )

Y2 = np.concatenate( (Y[:,0].T, Y[:,2].T, Y[:,5].T) )
p2 = dynamic.orthogonalize_p( Y2 )
for i in range(p2.shape[0]):
    p2[i,:] /= np.linalg.norm(p2[i,:])

print 'Difference with old method: %f' % np.linalg.norm(fs.p[0] - p2)

print '-'*60

###################################
# Clear and perform rejection
###################################

print 'Clearing with remove_mtx'

cl = np.matrix([[True, False, True, False, False, True],[False, False,
    False, False, False, False]])

fs.remove_mtx( cl )

print 'Updating with insert_mtx'

up = np.matrix([[True, True, True, False],
[True, True, False, True],
[True, False, True, True],
[False, True, True, True]])

fs.insert_mtx( up )

print '\nPerforming matrix rejection'
V = np.matrix([[1, 1, 1, 1],[1, 1, -1, -1],[1, -1, -1, 1],[1, -1, 1, -1]])

V2 = fs.reject_mtx( V )

print 'Residual inner products:'
print V2[0,:] * fs.p[0].T
print V2[1,:] * fs.p[1].T
print V2[2,:] * fs.p[2].T
print V2[3,:] * fs.p[3].T
