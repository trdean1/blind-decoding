import numpy as np
import scipy
import itertools
import pickle
from scipy import linalg,matrix
import math
import sys
import Xmats
from FeasibleRegion import FeasibleRegion

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


def is_feasible(U,Y,tol=1e-14,mask=[],prod=[]):
    '''
    Returns true if the given value of U is feasible.  Ignores entires
    in the product U*Y where mask is set to one.  This is done to ignore entries
    that correspond to active constraints --- numerical instability might lead
    to our point being just outside the feasible region
    '''
    if prod == []:
        prod = U*Y

    if mask == []:
        mask = np.matrix(np.ones(prod.shape))
    prod = np.multiply(mask,prod)

    if np.max(abs(prod)) > 1 + tol:
        return False
    else:
        return True


def get_active_constraints_bool(U,Y,tol=1e-9, prod=[]):
    '''
    Returns a boolean matrix where each entry tells whether or
    not the constraint is active (within tol)
    '''
    if prod == []:
        prod = U*Y
    pos = abs(1 - prod) < tol
    neg = abs(1 + prod) < tol

    #pos: constraint is met in the positive direction: <u_i, y_j> = +1
    #neg: constraint is met in the negative direction: <u_i, y_j> = -1
    return np.logical_or(pos,neg)

def get_distance_from_boundary(U,Y,tol=1e-2, X = []):
    if X == []:
        X = U*Y
    d = np.matrix(np.zeros( X.shape ) )

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if abs( abs(X[i,j]) - 1) < tol:
                d[i,j] = abs( abs(X[i,j]) - 1 )

    return d

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


def find_infeasible_point(U,V,Y,mask=[],UY = []):
    #finds a (non-optimal) value of t so that U+t*V is infeasible
    if not is_feasible(U,Y,1e-12,mask, UY):
        raise ValueError("Starting point not feasible")
    t = 1
    try: 
        while is_feasible(U+t*V,Y,1e-12,mask, UY):
            t *= 2
    except OverflowError:
        #print "t = %d\n" % t
        raise OverflowError

    return t


def binary_search(U,V,Y,t_start=0.0,t_stop=[],steps=64,mask=[], UY = []):
    #start must be feasible and stop must be infeasible
    if t_stop == []:
        t_stop = find_infeasible_point(U,V,Y,mask,UY=UY)

    if not is_feasible(U+t_start*V,Y,mask=mask,prod=UY):
        raise ValueError("Starting point not feasible")
    if is_feasible(U+t_stop*V,Y,mask=mask,prod=UY):
        raise ValueError("Stopping point is feasible")

    step = (t_stop - t_start) / 2.0
    t = t_start + step

    for i in range(steps):
        step /= 2
        if is_feasible(U+t*V,Y,1e-15,mask,prod=UY):
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


def row_to_vertex( U, feasible_region, row, UY = [], verbose=False ):
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

    
    if verbose:
        print "Attempting row %d" % row
        print "Bad indices: %s" % bad_index

    i = 0
    while not is_zero_one.all():
        bad = 0
        while i < n:
            y = np.matrix( np.copy( Y[:,bad_index[bad]] ) )
            v = np.matrix(np.zeros( (n,1) ) )
            v = v.T

            # Pick a random vector in the nullspace
            tries = n
            while abs(v*y) < 1e-9:
                tries -= 1
                if tries == 0:
                    if verbose:
                        print "Warning: Failed to find vector in nullsapce"
                        print u*Y
                    return u
                    #raise ValueError("Failed to find vector in nullspace")

                v = random_unit(n)
                v = feasible_region.reject_vec( v, row )
                norm = np.linalg.norm(v)

                if norm < 1e-13:
                    continue

                v = v / np.linalg.norm(v)

            #Force to +1
            t = (1 - np.dot( u, y ) ) / np.dot(v,y)

            #this condition implies that we are already at an edge in this
            #direction.  Continue to the next subspace if needed
            if abs(t) < 1e-6:
                i += 1
                continue

            else:
                u_new = u + t*v

                #If u_new is not feasible, try forcing to -1 instead of +1
                if not is_feasible(u_new,Y,tol=1e-5):
                    t = (-1 - np.dot( u, y ) ) / np.dot(v,y)
                    u_new = u + t*v 
                    if not is_feasible(u_new,Y,tol=1e-5):
                        #try picking a different target in this row, this face
                        #might be infeasible
                        bad += 1
                        if bad == len(bad_index):
                            #something went wrong, problem could be singular                 
                            #raise ValueError('Failed to find a vertex')
                            if verbose:
                                print "Warning: Failed to find a vertex"
                                print "Row = %d, i = %d" % (row, i)
                                print uY
                            return u
                        else:
                            continue

                #We activated another constraint, save values and leave
                u = u_new

                i = i + 1
                uY = u * Y
                is_zero_one = check_zero_one( uY )
                is_zero_one_list = is_zero_one.tolist()[0] 

                #Update the newly active constraints
                for ii,v in enumerate(is_zero_one_list):
                    if v:
                        feasible_region.insert(row, ii)

                #indices of all the bad entries
                bad_index = [ii for ii, j in enumerate(is_zero_one_list) if not j ]

                break

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


def find_vertex_on_face(U, Y, feasible_region, UY = [], verbose=False):
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

        U[i,:] = row_to_vertex(U, feasible_region, i, UY, verbose)
        

    return U


def dynamic_solve(U,Y, verbose=False):
    (n,k) = Y.shape
    V = objgrad(U)
    t = binary_search(U,V,Y)
    U = U+t*V

    p_bool = np.matrix(np.zeros(Y.shape)) == 1
    fr = FeasibleRegion( Y )
    p = []

    for i in range(n**2 - 1):
        XX = U*Y

        if verbose:
            print '\n%s' % ('-'*40)
            print "Iteration %d" % i
            print "U*Y = %s" % XX
            print "Error = %s" % get_distance_from_boundary( U,Y )
            
        p_bool_new = get_active_constraints_bool(U,Y,prod=XX)
        p_update = p_bool_new ^ p_bool
        p_bool = p_bool_new

        fr.insert_mtx( p_bool )
        
        V = objgrad(U)
        if verbose:
            print "Gradient: "
            print V

        V = fr.reject_mtx( V )

        if np.linalg.norm(V) < 1e-12:
            # XXX: This means we aren't at a vertex but the graident is
            # has no component that lies in the nullspace.  It should 
            # be possible to handle this by picking V at random and rejecting
            # it from p.  There's often some numerical stability issues here and
            # so I handle this with the function find_vertex_on_face
            if verbose:
                print "Gradient is orthogonal to null space at step %s" % i
            if not check_zero_one( XX ).all():
                if verbose:
                    print "Calling find_vertex_on_face"
                U = find_vertex_on_face(U,Y, fr, XX, verbose)

            break 

        mask = np.logical_not(get_active_constraints_bool(U,Y,tol=1e-8, prod=XX))*1
        #t = binary_search(U,V,Y,mask=mask)
        t = boundary_dist(U, V, Y, XX, mask)
        U = U+t*V
        if verbose:
            print 'V = %s' % V
            print 't = %s' % t
            print '%s' % ('-'*40)

    if verbose:
        print "U * Y = %s" % (U*Y)


    return U

def trial(n,k,verbose=False):
    #Generate channel and observed samples
    A = np.random.randn(n,n)
    X = Xmats.X_guarantee(n, k)
    Y = A*X
    k = Y.shape[1]

    Ui = rand_init(n,Y)
    try:
        U = dynamic_solve(Ui,Y, verbose)
    except: 
        instance = { 'U' : Ui,
                     'Y' : Y }
        pickle.dump( instance, open("error.p", "wb") )
        raise 
        

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
