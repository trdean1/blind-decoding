import numpy as np
import scipy
import itertools
import pickle
from scipy import linalg,matrix
import math
import sys

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


def rand_init(n,Y):
    #Generate initial value
    U = random_orthogonal(n)
    scale = 1
    while not is_feasible(U,Y):
        U = random_orthogonal(n)
        U /= scale
        scale = scale*2

    return U

def row_to_vertex( u, Y ):
    '''
    This function is highly redundant.  We just need to pull a specific 
    entry of p out and then pick a random direction in the null space of 
    of the result.  There isn't a good way to do this so we just make
    p from scratch and call SVD


    u*Y should have some good and some bad entries
    will move u until u*Y has all 'good' entries
    may still be zeros but shouldn't matter in end
    '''
    print "Starting uY = %s\n" % (u*Y)
    print "Y = %s\n" % Y
    n = Y.shape[0]
    is_zero_one = check_zero_one( u*Y )
    is_zero_one_list = is_zero_one.tolist()[0] 

    #indices of all the bad entries
    bad_index = [i for i, j in enumerate(is_zero_one_list) if not j ]

    while not is_zero_one.all():
        #Pull out the 'good' columns of Y
        Y_space = Y
        dropped = 0
        for i,col in enumerate(is_zero_one_list):
            if not col:
               Y_space = np.delete(Y_space,i-dropped,axis=1)
               dropped += 1
               #'Target' entry is just last bad value we come across
               #y = Y[:,i]

        #print "y = %s" % Y[:,bad_index[0]]
        #Y_perp = nullspace(Y_space.transpose())
        while True:
            #Check if this has a left null space
            Y_perp = nullspace(Y_space.transpose())
            if Y_perp.shape == (1,0):
                #if no nullspace, drop a column with a zero
                is_zero = abs(uY) <= 1e-12
                deleted = False
                for i,bb in enumerate(is_zero):
                    if bb:
                        Y_space = np.delete(Y_space, i, axis=1)
                        deleted = True
                        break

                if not deleted: 
                    #If we have no columns to delete, bail
                    raise ValueError

            #Found a nullspace so we can use it
            else:
                break

        print "Y_space = %s" % Y_space
        print "Y_perp = %s" % Y_perp

        #print "Y_perp = %s\n" % Y_perp
        i = 0
        bad = 0
        while i < Y_perp.shape[1]:
            y = Y[:,bad_index[bad]]
            v = Y_perp[:,i] #This is an arbitrary direction in the null space
            #move zig
            if np.dot( v,y ) == 0:
                #This should never happen since v \in y^\perp
                raise ValueError

            t = (1 - np.dot( u, y ) ) / np.dot(v,y)
            #this condition implies that we are already at an edge in this
            #direction
            if abs(t) < 1e-6:
                i += 1
                continue

            else:
                print "uy_good = %s" % (u*Y_space)
                print "uy_bad = %s" % (u*y) 
                print "t = %s\nv = %s" % (t, v)
                print "(u+t*v)Y = %s" % ((u+t*v)*Y)
                u_new = u + t*v
                if not is_feasible(u_new,Y,tol=1e-5):
                    print "Infeasible, trying -1"
                    t = (-1 - np.dot( u, y ) ) / np.dot(v,y)
                    u_new = u + t*v 
                    print "u = %s\nt = %s\nv = %s" % (u, t, v)
                    print "(u+t*v)Y = %s" % ((u+t*v)*Y)
                    if not is_feasible(u_new,Y,tol=1e-5):
                        #try picking a different target in this row, this face
                        #might be infeasible
                        if bad < len(bad_index):
                            bad += 1
                            continue
                        else:
                            #something is very infeasible or wrong here!
                            raise ValueError

                #print "u = %s\nt = %s\nv = %s" % (u, t, v)
                #print "(u+t*v)Y = %s" % ((u+t*v)*Y)

                u = u_new
                is_zero_one = check_zero_one( u*Y )
                is_zero_one_list = is_zero_one.tolist()[0] 
                #indices of all the bad entries
                bad_index = [i for i, j in enumerate(is_zero_one_list) if not j ]
                break

        #Not sure if we need to do something here...we should be close to \pm 1
        #if i == Y_perp.shape[1]:
        #    raise RuntimeError

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


def find_vertex_on_face(U, Y, UY = [], p = []):
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
        if UY == []:
            UY = U*Y

        is_zero_one = check_zero_one( UY )

        #Bail if all entries are already in {-1,0,1}
        if is_zero_one.all():
            break

        #We want to find the first row with an entry not in {-1, 0, 1}
        if (is_zero_one[i,:]).all():
            continue

        #if p == []:
        #TODO: need to take advantage of p being precomputed.

        U[i,:] = row_to_vertex(U[i,:], Y)
        

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
    p = []

    for i in range(n**2 - 1):
        print '\n%s' % ('-'*40)
        print "Iteration %d" % i
        XX = U*Y
        print "U*Y = %s" % XX

        p_bool_new = get_active_constraints_bool(U,Y)
        p_update = p_bool_new ^ p_bool
        p_bool = p_bool_new
        #print "Update = %s" % p_update
        p = update_p(p, p_update, Y)
        orthogonalize_p(p)

        #p = get_active_constraints_basis(U,Y)
        #p = p / np.linalg.norm(p)
        V = objgrad(U)
        v_vec = np.reshape(V,[n**2,1])
        #v_vec = project_to_null(v_vec,p.transpose())

        n_constraints = p.shape[0]
        print "Iteration %d has %d independent constraints" % (i,n_constraints)
        Xt = U*Y

        for i in range(n_constraints):
            s = (np.dot(p[i,:],v_vec)/np.dot(p[i,:],p[i,:].transpose()))
            v_vec = v_vec.transpose() - s*p[i,:]
            v_vec = v_vec.transpose()

        if np.linalg.norm(v_vec) < 1e-12:
            print "Gradient is orthogonal to null space at step %s" % i
            if not check_zero_one( Xt ).all():
                print "Need to call find_vertex_on face"
                U = find_vertex_on_face(U,Y, Xt)
            #print "UY = %s\n" % (U*Y)
            break 

        V = np.reshape(v_vec, [n,n])
        mask = np.logical_not(get_active_constraints_bool(U,Y))*1
        t = binary_search(U,V,Y,mask=mask)
        #mask = get_active_constraints_bool(U,Y)*100
        #diff = abs((1-abs(U*Y) + mask)/(V*Y + mask)) + mask
        #t = diff.min()
        U = U+t*V
        print '%s' % ('-'*40)

    print "U * Y = %s" % (U*Y)

    #if not check_zero_one( U*Y ).all():
    #    try:
    #        print '%s' % ('-'*40)
    #        U = find_vertex_on_face_new(U, Y)
    #        print "U * Y = %s" % (U*Y)
    #    except RuntimeError:
    #        print "Warning: Weird RuntimeError"
    #        pass

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



def examine_instances(instances): 
    for i, instance in enumerate(instances):
        if i == 1:
            continue
        print "\n***BEGIN INSTANCE %d***\n%s\n" %(i, '-' * 40)
        print "X = \n%s\n" %(instance['X'])
        print "A = \n%s\n" %(instance['A'])
        print "Ui = \n%s\n" %(instance['Ui'])
        print "U = \n%s\n" %(instance['U'])
        print "Y = \n%s\n" %(instance['Y'])
        print "UY = \n%s\n" %(instance['UY'])
        print "\n***END INSTANCE %d***\n%s\n" %(i, '-' * 40)
        U = dynamic_solve(instance['Ui'],instance['Y'])
        UY = U * instance['Y']
        print "U = \n%s\n" %(U)
        print "UY = \n%s\n" %(UY)
        print "UA = \n%s\n" % (U*instance['A'])
        print "det UA = %s\n" % (np.linalg.det(U*instance['A']))


def load_bad_instances(bad_instance_file): 
    with open(bad_instance_file, 'r') as fp:
        unpickler = pickle.Unpickler(fp)
        bad_instances = unpickler.load()
        examine_instances(bad_instances)

######################################################################
# Ignore these functions, they were from another idea I previously had
######################################################################

def givens_rotation_full(i,j,theta,n):
    if j < i:
        tmp = i
        i = j
        j = tmp

    G = np.matrix(np.eye(n))
    c = math.cos(theta)
    s = math.sin(theta)
    G[i,i] = c
    G[j,j] = c
    G[i,j] = s
    G[j,i] = -1*s

    return G

def givens_rotation_part(s11,s22,s12):
    #Should be able to simplify this if needed
    theta = math.pi/4
    if abs(s11-s22) > 1e-6:
        at = 2*s12/(s22-s11)
        theta = math.atan(at)/2
    c = math.cos(theta)
    s = math.sin(theta)
    r = math.hypot(s11,s12) #Not sure we actually need to return this
    return (c, s, r)

def apply_rotation(R,c,s,i,j):
    '''
    Should be able to save more from this because R is sparse
    '''
    (n,m) = R.shape

    for k in range(m):
        rik = c*R[i,k]-s*R[j,k]
        rjk = c*R[j,k]+s*R[i,k]
        R[i,k] = rik
        R[j,k] = rjk

def conjugate_rotation(A,c,s,i,j):
    '''
    Assumes A is symmetric
    '''
    a11 = A[i,i]
    a22 = A[j,j]
    a12 = A[i,j]
    A[i,i] = (c**2)*a11-2*c*s*a12+(s**2)*a22
    A[j,j] = (s**2)*a11+2*c*s*a12+(c**2)*a22
    A[i,j] = a12*(c**2 - s**2) + s*c*(a11 - a22)
    A[j,i] = A[i,j]



#load_bad_instances('bad_instances')

