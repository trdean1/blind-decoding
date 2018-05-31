import numpy as np
import scipy
import itertools
import pickle
from scipy import linalg,matrix
import math
import sys
from FeasibleRegion import FeasibleRegion

#Proper sets of samples of X so that we are guaranteed to recover A^-1 up to an ATM
bases = { 2 : np.matrix([[1,1],[1,-1]]),\
        3 : np.matrix([[1,1,1],[1,-1,1],[-1,1,1]]),\
        4 : np.matrix([[1,1,1,1,1],[1,1,-1,-1,1],[1,-1,1,-1,1],[1,-1,-1,1,-1]]),\
        5 : np.matrix([[-1,-1,-1,-1,-1,-1],[-1,-1,-1,1,1,-1],[-1,-1,1,-1,1,1],[-1,1,-1,-1,1,1],[-1,1,1,1,-1,-1]])}

def objfunc(x):
    '''
    Input: n x n numpy matrix of doubles
    Output: double f = log | det (x) |
    '''
    return np.log(abs(np.linalg.det(x)))
    #return np.linalg.norm(x,2)


def objgrad(x):
    '''
    Input: n x n numpy matrix of doubles
    Output: nxn matrix of doubles, in this case the inverse of x transposed
    '''
    return np.linalg.inv(x).transpose()

def is_feasible(U,Y,tol=1e-14,mask=[]):
    '''
    Returns true if the given value of U is feasible.  Ignores entires
    in the product U*Y where mask is set to one.  This is done to ignore entries
    that correspond to active constraints --- numerical instability might lead
    to our point being just outside the feasible region
    '''
    prod = U*Y

    if mask == []:
        mask = np.matrix(np.ones(prod.shape))
    prod = np.multiply(mask,prod)

    if np.max(abs(prod)) > 1 + tol:
        return False
    else:
        return True

def get_active_constraints_bool(U,Y,tol=1e-9):
    '''
    Returns a boolean matrix where each entry tells whether or
    not the constraint is active (within tol)
    '''
    prod = U*Y
    pos = abs(1 - prod) < tol
    neg = abs(1 + prod) < tol

    #pos: constraint is met in the positive direction: <u_i, y_j> = +1
    #neg: constraint is met in the negative direction: <u_i, y_j> = -1
    return np.logical_or(pos,neg)

def get_active_constraints_basis(U,Y,tol=1e-9):
    '''
    Returns a basis of the constraints that are currently active.
    A constraint is active if U*Y is within tol of 1.  
    ''' 
    c_bool = get_active_constraints_bool(U,Y,tol)
    (n,k) = c_bool.shape
     
    basis = []
    for i in range(n):
        for j in range(k):
            if c_bool[i,j]:
                row = np.matrix(np.zeros([1,n*n]))
                row[0,i*n:(i+1)*n] = y[:,j].transpose()
                if basis == []:
                    basis = row
                else:
                    basis = np.concatenate((basis,row))

    return basis

def random_orthogonal(n):
    '''
    Input: positive integer n
    Output: a random nxn matrix with determinant 1
    '''

    v = np.matrix(np.random.rand(n,n))
    (q,r) = np.linalg.qr(v)

    return q

    return v.compress(mask,axis=0)

def random_unit(n):
    v = np.matrix(np.random.rand( n,1 ) )
    return v / np.linalg.norm(v)

def nullspace(A,eps=1e-12):
    '''
    Returns a basis of the null space of the matrix A, with a tolerance given by
    eps.  I.e., singular vectors whose singular values are less than eps are
    considered part of the null space
    '''
    u,s,vh = scipy.linalg.svd(A)
    padding = max(0,np.shape(A)[1]-np.shape(s)[0])
    null_mask = np.concatenate(((s<=eps),np.ones((padding,),dtype=bool)),axis=0)
    null_space = scipy.compress(null_mask,vh,axis=0)
    return scipy.transpose(null_space)


def project_to_null(v,A):
    '''
    projects the vector v onto the nullspace of A
    '''
    #These elements of A are all zeros
    zeros = (np.sum(A,1) == 0)*1
    if np.sum( zeros ) == 0:
        #print "Using SVD to find nullspace"
        #If we have no rows of A that are all zero, find nullspace through SVD
        null = nullspace(A.transpose())
        null = np.matrix(null).transpose()
        rows,cols = null.shape
        r = np.zeros(v.shape)
        for row in range(rows):
            r += (null[row] * v).item() * null[row].transpose()
    else:
        #print "Simple zero projection"
        r = np.multiply( v, zeros )


    #print "Projection onto null:\n %s\n" % r
    return r

def find_infeasible_point(U,V,Y,mask=[]):
    #finds a (non-optimal) value of t so that U+t*V is infeasible
    if not is_feasible(U,Y,1e-12,mask):
        p = U*Y
        #print "U*Y=%s\n" % p
        raise ValueError("Starting point not feasible")
    t = 1
    try: 
        while is_feasible(U+t*V,Y,1e-12,mask):
            t *= 2
    except OverflowError:
        #print "t = %d\n" % t
        raise OverflowError

    return t

def binary_search(U,V,Y,t_start=0.0,t_stop=[],steps=64,mask=[]):
    #start must be feasible and stop must be infeasible
    if t_stop == []:
        t_stop = find_infeasible_point(U,V,Y,mask)

    if not is_feasible(U+t_start*V,Y,mask=mask):
        raise ValueError("Starting point not feasible")
    if is_feasible(U+t_stop*V,Y,mask=mask):
        raise ValueError("Stopping point is feasible")

    step = (t_stop - t_start) / 2.0
    t = t_start + step

    for i in range(steps):
        step /= 2
        if is_feasible(U+t*V,Y,1e-12,mask):
            t += step
        else:
            t -= step

    return t

def boundary_dist(U, V, Y, uy = [], mask = []): #{{{
    '''
    Calculate the distance to the problem boundary along a given vector.
    Input:  U = current feasible solution
            V = vector along which to travel
            Y = received set of symbols
            mask = points to consider when calculating feasibility, by default
                all points are considered
    Output: t = maximum distance such that U + t * V remains feasible
    '''
    # Calculate U * Y and V * Y once.
    if uy == []:
        uy = U * Y

    dy = V * Y

    # Find the lowest value of t such that U + t * V reaches the boundary.
    t_min = None
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if not mask == [] and not mask[i, j]:
                continue
            # Determine the value of t such that the [i, j] constraint reaches
            # the boundary.
            if dy[i, j] < 0:
                t = (-1 - uy[i, j]) / dy[i, j]
            elif dy[i, j] > 0:
                t = (1 - uy[i, j]) / dy[i, j]
            else:
                t = None

            if not t is None and (t_min is None or t < t_min):
                t_min = t

    return t_min

def rand_init(n,Y):
    #Generate initial value
    U = random_orthogonal(n)
    scale = 1
    while not is_feasible(U,Y):
        U = random_orthogonal(n)
        U /= scale
        scale = scale*2

    return U

def row_to_vertex( U, feasible_region, row, UY = [] ):
    '''
    This function is highly redundant.  We just need to pull a specific 
    entry of p out and then pick a random direction in the null space of 
    of the result.  There isn't a good way to do this so we just make
    p from scratch and call SVD


    u*Y should have some good and some bad entries
    will move u until u*Y has all 'good' entries
    may still be zeros but shouldn't matter in end
    '''

    #Check if we got a precomputed UY
    uY = None
    if UY == []:
        uY = u * feasible_region.Y
    else:
        uY = UY[row,:]

    u = U[row,:]
    Y = feasible_region.Y
    n = feasible_region.n

    #Get indices of all entries not in {-1,0,1}
    is_zero_one = check_zero_one( uY )
    is_zero_one_list = is_zero_one.tolist()[0] 
    bad_index = [i for i, j in enumerate(is_zero_one_list) if not j ]

    #print UY
    #print "Trying to fix row %d" % row
    #print "Bad indices: %s" % bad_index

    i = 0
    while not is_zero_one.all():
        ###############################################################
        #TODO: Triple check that this loop below will never do anything
        # row_to_vertex can only get called if the gradient is
        # in the nullspace of the active constraints, so p will
        # never be full rank unless something went wrong...
        #
        #while True:
        #    #Check if this has a left null space
        #    #I actually don't think this code can ever be called
        #    #
        #
        #    # Check if feasbile region has a left nullspace
        #    if feasible_region.p[row].shape[0] == n:
        #        #if no nullspace, drop a column with a zero
        #        is_zero = abs(uY) <= 1e-12
        #        deleted = False
        #        for i,bb in enumerate(is_zero):
        #            if bb:
        #                #Y_space = np.delete(Y_space, i, axis=1)
        #                feasible_region.remove( row, i )
        #                deleted = True
        #                break
        #
        #        if not deleted: 
        #            #If we have no columns to delete, bail
        #            raise ValueError
        #
        #    #Found a nullspace so we can use it
        #    else:
        #        break
        ################################################################

        Y_space = feasible_region.b[row].T
        if i >= n:
            print "\n\nFailed with i=%d, bad=%d" %(i, bad)
            print bad_index
            print feasible_region
            print u*Y
            raise RuntimeError("This again")

        bad = 0
        while i < n:
            #print "Bad entries: %s" % bad_index
            #print "i = %d Target %d" % (i, bad_index[bad])
            try:
                y = np.matrix( np.copy( Y[:,bad_index[bad]] ) )
            except IndexError:
                print "Badness here:"
                print "Y = %s" % Y
                print "Bad indices: %s" % bad_index
                print "Index: %d" % bad
                raise IndexError
            v = np.matrix(np.zeros( (4,1) ) )
            v = v.T

            # Pick a random vector in the nullspace
            tries = 10
            while abs(v*y) < 1e-9:
                v = random_unit(n)
                v = feasible_region.reject_vec( v, row )
                norm = np.linalg.norm(v)
                if norm < 1e-13:
                    tries -= 1
                    if tries == 0:
                        raise ValueError('Constraints appear full rank')
                    continue

                v = v / np.linalg.norm(v)

            t = (1 - np.dot( u, y ) ) / np.dot(v,y)

            #this condition implies that we are already at an edge in this
            #direction.  Continue to the next subspace if needed
            if abs(t) < 1e-6:
                i += 1
                #print "Already edge in this direction"
                continue

            else:
                u_new = u + t*v

                #If u_new is not feasible, try forcing to -1 instead of +1
                if not is_feasible(u_new,Y,tol=1e-5):
                    #print "Trying -1"
                    t = (-1 - np.dot( u, y ) ) / np.dot(v,y)
                    u_new = u + t*v 
                    if not is_feasible(u_new,Y,tol=1e-5):
                        #try picking a different target in this row, this face
                        #might be infeasible
                        bad += 1
                        if bad == len(bad_index):
                            #something went wrong, problem could be singular                 
                            #raise ValueError('Failed to find a vertex')
                            print "Warning: Failed to find a vertex"
                            print "Row = %d, i = %d" % (row, i)
                            print uY
                            return
                        else:
                            continue

                #We activated another constraint, save values and leave
                u = u_new
                i = i + 1
                #print "Found vertex"
                #TODO: This code seems to have no function?
                uY = u * Y
                #print uY                
                is_zero_one = check_zero_one( uY )
                is_zero_one_list = is_zero_one.tolist()[0] 
                for ii,v in enumerate(is_zero_one_list):
                    if v:
                        feasible_region.insert(row, ii)
                #indices of all the bad entries
                bad_index = [ii for ii, j in enumerate(is_zero_one_list) if not j ]
                #print is_zero_one_list

                #TODO: Is break the right call here?
                break

        #Not sure if we need to do something here...we should be close to \pm 1
        #if i == n:
        #   print uY
        #    print abs(uY) - 1
        #    raise RuntimeError("This didn't work?")

        if not is_feasible(u, Y,tol=1e-5):
            print u * Y
            print feasible_region
            raise RuntimeError 

    return u

def check_zero_one(A,eps=1e-6):
    '''
    Return a boolean matrix the same size as A.  
    b_ij is true if a_ij \in {-1,0,1}
    '''
    is_one = abs(abs(A) - 1) <= eps
    is_zero = abs(A) <= eps
    is_one_zero = np.logical_or( is_one, is_zero )

    return is_one_zero


def find_vertex_on_face(U, Y, feasible_region, UY = []):
    '''
    U must be on a face of the polytope that bounds feasible region.
    This face must have constant slope.  Then this function will move to a
    vertex of that face. UY may still have zero entries if k > n

    Pass in precomputed UY to avoid recomputing
    Pass in p to avoid calling SVD to find nullspace
    '''

    # Assuming U is a matrix, then n is number of rows
    n = np.shape(U)[0]
    
    #go row by row
    for i in range(n):
        if UY == [] or i != 0:
            UY = U*Y

        is_zero_one = check_zero_one( UY )

        #Bail if all entries are already in {-1,0,1}
        if is_zero_one.all():
            break

        #We want to find the first row with an entry not in {-1, 0, 1}
        if (is_zero_one[i,:]).all():
            continue

        U[i,:] = row_to_vertex(U, feasible_region, i, UY)
        

    return U

def update_p( p, update, Y ):
    '''
    For any entry in update that is True, append the appropriate row to p
    '''
    (n,k) = Y.shape
    for i in range(n):
        for j in range(k):
            if update[i,j] != True:
                continue
            
            row = np.matrix(np.zeros([1,n*n]))
            row[0,i*n:(i+1)*n] = Y[:,j].transpose()
            if p == []:
                p = row
            else:
                p = np.concatenate((p,row))

    return p

def orthogonalize_p(p,tol=1e-9):
    '''
    Converts p into an orthogonal basis.  This is basically an optimized QR
    decomposition
    '''
    #This multiply can be made faster since p is sparse
    #also notice that we update one row of p at time so we should
    #only need to do a rank one update of A
    A = p * p.T
    n = A.shape[0]

    #Find non-zero entires in A, should be easier using a sparse
    #implementation
    bad = []
    for i in range(n):
        for j in range(n):
            if j <= i:
                continue
            if abs(A[i,j]) > tol:
                if not ( i in bad ):
                    bad.append(i)
                if not ( j in bad ):
                    bad.append(j)

    #print "Non-orthogonal vectors: %s" % bad

    zero = []
    #Most of the time, not all vectors overlap all other vectors
    #so there might be some optimization by not iterating over all pairs 
    #However, worst case we will have k elements in bad and k(k-1) pairs.
    #Most commonly we have 2 elements in bad so it doesn't matter
    for (i,j) in itertools.combinations(bad,2):
        if (i in zero) or (j in zero):
            continue
        #Perform vector rejection
        u = p[i,:]
        v = p[j,:]
        vv = v - ((u*v.T)/(u*u.T))*u
        if np.linalg.norm(vv) < tol:
            if not (j in zero):
                zero.append(j)
        else:
            p[j,:] = vv / np.linalg.norm(vv)

    zero.sort()
    #print "Redundant contraints: %s" % zero
    p = np.delete(p,zero,0)

    #A = p * p.T
    #A = np.round(100*A)/100.0
    #print "p * p.T =\n%s" % A
    return p

def dynamic_solve(U,Y):
    (n,k) = Y.shape
    V = objgrad(U)
    #diff = abs((1-abs(U*Y))/(V*Y))
    t = binary_search(U,V,Y)
    #t = diff.min()
    U = U+t*V

    p_bool = np.matrix(np.zeros(Y.shape)) == 1
    fr = FeasibleRegion( Y )
    p = []

    for i in range(n**2 - 1):
        #print '\n%s' % ('-'*40)
        #print "Iteration %d" % i
        XX = U*Y
        #print "U*Y = %s" % XX

        p_bool_new = get_active_constraints_bool(U,Y)
        p_update = p_bool_new ^ p_bool
        p_bool = p_bool_new

        fr.insert_mtx( p_bool )
        
        V = objgrad(U)
        V = fr.reject_mtx( V )

        if np.linalg.norm(V) < 1e-12:
            #print "Gradient is orthogonal to null space at step %s" % i
            if not check_zero_one( XX ).all():
                #print "Need to call find_vertex_on_face"
                U = find_vertex_on_face(U,Y, fr, XX)

            break 

        mask = np.logical_not(get_active_constraints_bool(U,Y))*1
        t = binary_search(U,V,Y,mask=mask)
        #t = boundary_dist(U, V, Y, XX, mask)
        U = U+t*V
        #print '%s' % ('-'*40)

    #print "U * Y = %s" % (U*Y)

    return U

def rand_zero_one( n, k, use_basis=True ):
    if use_basis:
        X = bases[n]
        (nn,kk) = X.shape
        if kk < k:
            XX = np.matrix( 2*np.random.randint( 0, 2, size=(n,k-kk) ) - 1 )
            X = np.concatenate( (X, XX), axis=1 )
        return X
    else:
        return np.matrix( 2*np.random.randint( 0, 2, size=(n,k) ) - 1 )

def trial(n,k):
    #Generate channel and observed samples
    A = np.random.randn(n,n)
    #X = bases[n]
    X = rand_zero_one( n, k )
    Y = A*X
    k = Y.shape[1]

    Ui = rand_init(n,Y)
    U = dynamic_solve(Ui,Y)

    #if np.min(abs(U*Y)) > 0.99:
    #    print("Solved")
    #else:
        #print("Not quite")
        #print("U:")
        #print(U)
        #print("Y:")
        #print(Y)
        #U = dynamic_solve(U,Y)

    return U*Y
