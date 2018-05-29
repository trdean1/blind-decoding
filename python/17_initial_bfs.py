#!/usr/bin/env python
# imports{{{
import numpy as np
import scipy
from scipy import linalg, matrix
import sys
import pickle
import copy
import select
import parse_trap
import matplotlib.pyplot as plt
# end imports}}}

# Constants and principal functions.{{{
def equal_atm(A, B, zthresh = 1e-5): # {{{
    '''
    Determine if matrices A and B are equal up to an ATM.
    '''
    if A.shape != B.shape:
        return False

    used_rows = [False for i in range(B.shape[0])]
    for i in range(A.shape[0]):
        matched = False
        for k in range(len(used_rows)):
            if used_rows[k]:
                continue
            if np.max(np.abs(A[i] - B[k])) < zthresh \
                    or np.max(np.abs(A[i] + B[k])) < zthresh:
                matched = True
                used_rows[k] = True
        #end for
        if not matched:
            return False
    #end for

    return True
#end def # }}}
def mtx2hash(X): # {{{
    '''
    Create a python hashable string representation of a matrix.
    '''
    s = ''
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            s += '1' if X[i, j] == 1 else '0'
        # end for
    # end for

    return s
# end def}}}
def hash2mtx(s, n): # {{{
    '''
    Create a matrix from a given hash.
    '''
    ncols = len(s) / n
    X = np.zeros((n, ncols))
    for i in range(n):
        for j in range(ncols):
            if s[i*n + j] == '1':
                X[i, j] = 1
            else:
                X[i, j] = -1
            # end if
        # end for
    # end for

    return X
# end def}}}
# End constants and principal functions.}}}

# single trial internal functions{{{
def objfunc(x): #{{{
    '''
    Input: n x n numpy matrix of doubles
    Output: double f = log | det (x) |
    '''
    return np.log(abs(np.linalg.det(x)))
#end #}}}
def objgrad(x): #{{{
    '''
    Input:  n x n numpy matrix of floats
    Output: the inverse of x transposed [also n x n matrix of floats]
    '''
    return np.linalg.inv(x).transpose()
#end #}}}
def is_feasible(U, Y, tol=1e-14, mask=[]): #{{{
    '''
    Returns true if the given value of U is feasible.  Ignores entries
    in the product U*Y where mask is set to one.  This is done to ignore entries
    that correspond to active constraints -- numerical instability might lead
    to our point being just outside the feasible region.
    '''
    prod = U * Y

    if mask == []:
        mask = np.matrix(np.ones(prod.shape))
    prod = np.multiply(mask, prod)

    if np.max(abs(prod)) > 1 + tol:
        return False
    else:
        return True
#end #}}}
def get_active_constraints_bool(U, Y, tol=1e-9): #{{{
    '''
    Returns a boolean matrix where each entry tells whether or not the
        constraint is active (within tol)
    Input:  U = feasible solution (n x n)
            Y = matrix of received symbols (n x k)
    Output: Boolean matrix (n x k) where output[i, j] = True iff the ith row of
    U causes the jth constraint of U * Y to be active, i.e. if |<u_i, y_j| = 1.
    '''
    prod = U * Y
    pos = abs(1 - prod) < tol
    neg = abs(1 + prod) < tol

    #pos: constraint is met in the positive direction: <u_i, y_j> = +1
    #neg: constraint is met in the negative direction: <u_i, y_j> = -1
    return np.logical_or(pos, neg)
#end #}}}
def get_active_constraints_basis(U, Y, tol=1e-9): #{{{
    '''
    Returns a basis of the constraints that are currently active.
    A constraint is active if U*Y is within tol of 1.
    Input:  U = feasible solution (n x n)
            Y = matrix of received symbols (n x k)
    Output: A matrix (q x n^2) (q is the number of active constraints) where
    each row corresponds to an (i, j) such that |<u_i, y_j>| = 1, meaning that
    the jth constraint (j \in 0..k) on the ith row of U is active.  In this
    instance, the entries in columns i*n .. (i+1)*n will be contain (y_j)^T
    (they will have the jth received symbol), and all other entries will be 0.
    ''' 
    c_bool = get_active_constraints_bool(U, Y, tol)
    (n, k) = c_bool.shape
     
    basis = []
    for i in range(n):
        for j in range(k):
            # If the jth constraint is active on the ith row, then add a row to
            # the basis consisting of (y_j)^T set in columns i*n .. (i+1)*n.
            if c_bool[i, j]:
                row = np.matrix(np.zeros([1, n*n]))
                row[0, i*n : (i+1)*n] = Y[:,j].transpose()
                if basis == []:
                    basis = row
                else:
                    basis = np.concatenate((basis, row))

    return basis
#end #}}}
def random_orthogonal(n): #{{{
    '''
    Input: positive integer n
    Output: a random n x n matrix with determinant 1
    '''

    v = np.matrix(np.random.rand(n, n))
    (q, r) = np.linalg.qr(v)

    return q
#end #}}}
def nullspace(A, eps=1e-12): #{{{
    '''
    Returns a basis of the null space of the matrix A, with a tolerance given by
    eps.  I.e., singular vectors whose singular values are less than eps are
    considered part of the null space.
    '''
    # Compute SVD.
    u, s, vh = scipy.linalg.svd(A)
    # Need padding if the matrix A is not square.
    padding = max(0, np.shape(A)[1] - np.shape(s)[0])

    # Determine which right singular vectors are part of the nullspace of A.
    null_mask = np.concatenate(
            (
                (s <= eps),
                np.ones((padding, ), dtype = bool)
            ),
            axis = 0)
    null_space = scipy.compress(null_mask, vh, axis = 0)
    return scipy.transpose(null_space)
#end #}}}
def project_to_null(v, A): #{{{
    '''
    Projects the vector v onto the nullspace of A.
    '''
    null = nullspace(A.transpose())
    null = np.matrix(null).transpose()
    rows, cols = null.shape
    r = np.zeros(v.shape)
    for row in range(rows):
        r += (null[row] * v).item() * null[row].transpose()

    return r
#end #}}}
def boundary_dist(U, V, Y, mask = []): #{{{
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
#end #}}}
def rand_init(n, Y): #{{{
    '''
    Find a random feasible point, which is an n x n U such that all entries of U
    * Y are bounded by 1 in absolute value.
    Input:  Y = n x k matrix of received symbols, one symbol per column.
            n = number of elements per received symbol. 
    Output: U = n x n feasible inverse of the channel gain matrix.
    '''
    # Generate initial value: select U at random from Orthogonal(n).
    U = random_orthogonal(n)

    # While the randomly selected matrix is still infeasible, draw another
    # random orthogonal matrix and scale it down by a factor that increases
    # exponentially until we find a feasible U.
    scale = 1
    while not is_feasible(U, Y):
        U = random_orthogonal(n)
        U /= scale
        scale = scale * 2

    return U
#end #}}}
def find_bfs(U, Y, verbose = False): #{{{
    '''
    Find an intial BFS for the constraints |UY|_\infty <= 1.  Algorithm works by
    moving to the problem boundary, then following the projection of the
    gradient onto the nullspace of active constraints until hitting the boundary
    again.

    Input:  U = current estimate for inverse of channel gain matrix
            Y = n x k matrix of received signals
    Output: U = estimated inverse channel gain matrix that forms a BFS
    '''
    n = U.shape[0]

    # Compute U^{-1}^T, the gradient of the objective function.
    V = objgrad(U)
    # Move in direction of max gradient to the boundary of the feasible region.
    t = boundary_dist(U, V, Y)
    U = U + t * V

    for i in range(n**2 - 1):
        # Get basis for active constraints.
        p = get_active_constraints_basis(U, Y)
        p = p / np.linalg.norm(p)

        # Compute gradient from current solution and unroll it.
        V = objgrad(U)
        v_vec = np.reshape(V, [n**2, 1])

        # Project gradient onto nullspace of the current active constraints,
        # reform the projected gradient from an n^2 vec into an n x n matrix.
        v_vec = project_to_null(v_vec, p.transpose())
        if verbose:
            print "null projection norm = %s" %(np.linalg.norm(v_vec))
            print "det(U) = %s" %(np.linalg.det(U))
        V = np.reshape(v_vec, [n, n])

        # Calculate a move along projected gradient to the boundary of the
        # feasible region, which should activate 1 more constraint.  When
        # calculating feasibility mask out all active constraints.
        mask = np.logical_not(get_active_constraints_bool(U, Y)) * 1
        t = boundary_dist(U, V, Y, mask = mask)
        
        # If nowhere to go, just break.
        if t is None:
            if verbose:
                print "Did not find anywhere to go, breaking..."
            break

        # Actually perform the move along the projected gradient.
        U = U + t * V

        if verbose:
            num_active = np.sum(np.ones(Y.shape)[np.abs(U * Y) >= 1.0 - 1e-3])
            print "iter = %d\nnum_active = %d\nvalues:\n%s" \
                    %(i, num_active, U * Y)

    return U
# end}}}
def trial(X, verbose = False): #{{{
    '''
    Run the algorithm to find a nonsingular vertex of the feasible region.  In
    the current setup, the original transmitted symbol values X are selected
    such that all nonsingular vertices are both optima of the program and
    furthermore solutions to the blind decoding problem.

    Input:  X = matrix of transmitted symbols.
    Output: UY = Estimate of X using U generated by initial BFS finder.
            U = Estimate of A^-1 generated by initial BFS finder.
            Y = Matrix of received symbols.
            A = Ground truth channel gain matrix.
            Ui = Initial random feasible point.
    '''
    # Randomized n x n channel gain matrix.
    n = X.shape[0]
    A = np.random.randn(n, n)
    if verbose:
        print "A = \n%s\n" %(A)
    # Compute the received symbols Y, with no error.
    Y = A * X

    # Pick a random feasible point.
    Ui = rand_init(n, Y)
    # Find a nonsingular vertex of the feasible region.
    U = find_bfs(Ui, Y, verbose)

    # Calculate number of good cols of U * Y.
    # TODO: This is no longer necessary: done in FlexTab.
    UY = U * Y

    if verbose:
        print UY

    # Return the estimate of the symbols X, since U * Y = A^-1 * A * X = X.
    return UY, U, Y, A, Ui
#end }}}
# end function set}}}

# single specific trial functions{{{
def single_run(Xmats, verbose = False, errfile = None, trap = False): # {{{
    '''
    Run a single trial, selecting one matrix from Xmats at random, filling in
    the specified number of extra columns with iid \pm 1 entries.
    Xmats should be an array where each element is itself an array of 2
    elements:
        1. A base np.matrix
        2. An integer representing the number of extra columns to add to the
        base matrix, which will be iid \pm 1
    '''
    while True:
        selection = np.random.randint(len(Xmats))
        X, ncol = Xmats[selection]
        
        # Fill the extra columns with iid \pm 1 entries.
        if ncol > 0:
            X = np.concatenate((X, np.zeros([X.shape[0], ncol])), axis=1)
            X[:, -ncol:] = np.array(
                    np.random.rand(X.shape[0], ncol) < 0.5, dtype = np.int
                    ) * 2 - 1
        # end if

        UY, U, Y, A, Ui = trial(X)
        t = FlexTab(U, Y, verbose)
        if t.num_good_cols() < X.shape[0] + 1:
            continue

        try:
            t.solve()
        except Exception as e:
            if type(e) == ValueError and t.num_indep_cols() < t.n:
                print "EXCEPTION: %s: %s" %(e.message, e)
            continue

        # XXX: Deal specifically with trap instances.
        if trap:
            if not equal_atm(t.U, np.linalg.inv(A)) and t.is_trapped():
                return X, A, Y, Ui, U, t.U
            else:
                return None
        else:
            break # XXX: Clean this up a little.
    # end while

    # XXX: Refactor
    #print "COLS: good = %d, bad = %d, indep = %d" %(X.shape[1] - num_bad_cols,
    #        num_bad_cols, t.num_indep_cols())
    
    if equal_atm(t.U, np.linalg.inv(A)):
        print "EQUAL ATM"
        return 'EQATM'
    else:
        print "UNEQUAL"
        print "U =\n%s\nA^-1 =\n%s\n" %(t.U, np.linalg.inv(A))
        print "Y =\n%s" %(t.Y)
        print "U * Y = \n%s\n" %(np.round(t.U * Y))
        if np.max(np.abs(np.abs(t.U * Y) - 1)) < 1e-9:
            print "UYPM1"
            return 'UYPM1'
        if t.is_certain_max_5():
            print "UMAX"
            return 'UMAX'
        if t.is_trapped():
            print "TRAPPED"
            return 'TRAPPED'
        
        # This is an error, print some basics, save if specified.
        print "U = \n%s\nY = \n%s\nUi = \n%s\n" %(U, Y, Ui)
        if errfile is not None:
            with open(errfile, "w") as fp:
                pickler = pickle.Pickler(fp)
                pickler.dump([U, Y])
        # end if
        return 'ERROR'
# end def}}}
def repeat_run(X, A, Y, verbose = False, max_attempts = 100): # {{{
    attempts = 0
    while True:
        if attempts > max_attempts:
            return 'RUNOUT'
        attempts += 1

        # Obtain a new random feasible point.
        Ui = rand_init(X.shape[0], Y)
        # Find a nonsingular vertex of the feasible region.
        U = find_bfs(Ui, Y, verbose)
        t = FlexTab(U, Y, verbose)
        if t.num_good_cols() < X.shape[0] + 1: #XXX: Require +1
            continue
        try:
            t.solve()
        except Exception as e:
            if type(e) == ValueError and t.num_indep_cols() < t.n:
                print "EXCEPTION: %s: %s" %(e.message, e)
            continue
        break
    # end while

    if equal_atm(t.U, np.linalg.inv(A)) \
            or np.max(np.abs(np.abs(t.U * Y) - 1)) < 1e-9 \
            or t.n == 5 and t.is_certain_max_5():
        return 'SOLVED'
    elif t.is_trapped():
        return 'TRAPPED'
    else:
        return 'ERROR'
# end}}}
def flex_run(X, num_bad, verbose = False): # {{{
    '''
    Perform a single run of matrix X, requiring the number of bad columns
    returned by the initial BFS finder to be +num_bad+.
    Return True if we get an ATM.

    NOTE: Returns False anytime we do not get an ATM, even if we get a global
    optimum or some other potentially acceptable result.
    '''
    while True:
        UY, U, Y, A, Ui = trial(X)
        t = FlexTab(U, Y)
        num_bad_cols = X.shape[1] - t.num_good_cols()
        if num_bad_cols == num_bad:
            break
    # end while

    t.solve()

    print "A^-1 = \n%s\n" %(np.linalg.inv(A))
    if equal_atm(t.U, np.linalg.inv(A)):
        print "EQUAL ATM"
        return True
    else:
        print "UNEQUAL"
        return False
    # end if
# end def}}}

def n5(verbose = False): # {{{
    '''
    n=5 single run
    '''
    Xmats = [
        [ np.matrix([
           [-1,  -1,  -1,  -1,  -1,  1],
           [-1,  -1,  -1,   1,   1,  1],
           [-1,  -1,   1,  -1,   1,  1],
           [-1,   1,  -1,  -1,   1, -1],
           [-1,   1,   1,   1,  -1,  1],
        ]), 0],
        [ np.matrix([
           [-1,  -1,  -1,  -1,  -1,  1,  1],
           [-1,  -1,  -1,   1,   1,  1, -1],
           [-1,  -1,   1,  -1,   1,  1,  1],
           [-1,   1,  -1,  -1,   1, -1, -1],
           [-1,   1,   1,   1,  -1,  1,  1],
        ]), 0], 
        [ np.matrix([
            [-1,  -1,  -1,  -1,  -1],
            [-1,  -1,  -1,   1,   1],
            [-1,  -1,   1,  -1,   1],
            [-1,   1,  -1,  -1,   1],
            [-1,   1,   1,   1,  -1],
        ]), 5],
        # Top right
        [ np.matrix([
            [-1,  -1,  -1,  -1,  -1],
            [-1,  -1,   1,   1,   1],
            [-1,   1,  -1,  -1,   1],
            [-1,   1,  -1,   1,  -1],
            [ 1,  -1,  -1,   1,   1],
        ]), 2],
    ]
    
    return single_run(Xmats, verbose)
# end def}}}
def n5_baseplus(num_mtx = 5, reps_per_mtx = 100, verbose = False): # {{{
    '''
    n=5 single run with different bases.  Used to determine conditions that
    guarantee recovery up to an ATM.
    '''
    Xmats = [
        np.matrix([
           [-1,  -1,  -1,  -1,  -1],
           [-1,  -1,  -1,   1,   1],
           [-1,  -1,   1,  -1,   1],
           [-1,   1,  -1,  -1,   1],
           [-1,   1,   1,   1,  -1],
        ]),
        # Top right.
        np.matrix([
            [-1,  -1,  -1,  -1,  -1],
            [-1,  -1,   1,   1,   1],
            [-1,   1,  -1,  -1,   1],
            [-1,   1,  -1,   1,  -1],
            [ 1,  -1,  -1,   1,   1],
        ]),
    ]
    matrices = {}

    for _ in range(num_mtx):
        # Select a base matrix.
        X = Xmats[ np.random.randint(len(Xmats)) ]
        # Create extra columns as needed.
        ncol = np.random.randint(1, 4)
        X = np.concatenate((X, np.zeros([X.shape[0], ncol])), axis=1)
        X[:, -ncol:] = np.array(
                np.random.rand(X.shape[0], ncol) < 0.5, dtype = np.int
                ) * 2 - 1
        Xhash = mtx2hash(X)
        if not Xhash in matrices:
            matrices[Xhash] = [0, 0]

        for _ in range(reps_per_mtx):
            # Set up a matrix for single_run.
            res = single_run([[X, 0]], verbose)
            if res == 'EQATM':
                matrices[Xhash][0] += 1
            else:
                matrices[Xhash][1] += 1
        # end for
    # end for

    for mtxhash in matrices:
        X = hash2mtx(mtxhash, 5)
        ok, fail = matrices[mtxhash]
        print "%s\n[%d / %d]\n" %(X, ok, ok + fail) 
# end}}}
def n5_specific(reps_per_mtx = 100, verbose = False): # {{{
    Xmats = [ # {{{
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1,  1,  1, -1],
            [-1, -1,  1, -1,  1,  1],
            [-1,  1, -1, -1,  1,  1],
            [-1,  1,  1,  1, -1, -1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1,  1,  1,  1],
            [-1, -1,  1, -1,  1, -1],
            [-1,  1, -1, -1,  1,  1],
            [-1,  1,  1,  1, -1,  1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1,  1,  1,  1],
            [-1, -1,  1, -1,  1,  1],
            [-1,  1, -1, -1,  1, -1],
            [-1,  1,  1,  1, -1, -1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1,  1,  1, -1],
            [-1, -1,  1, -1,  1,  1],
            [-1,  1, -1, -1,  1,  1],
            [-1,  1,  1,  1, -1,  1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1,  1,  1,  1],
            [-1, -1,  1, -1,  1, -1],
            [-1,  1, -1, -1,  1,  1],
            [-1,  1,  1,  1, -1, -1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1,  1,  1,  1],
            [-1, -1,  1, -1,  1,  1],
            [-1,  1, -1, -1,  1, -1],
            [-1,  1,  1,  1, -1,  1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1,  1,  1, -1],
            [-1,  1,  1, -1,  1, -1],
            [-1, -1,  1,  1, -1,  1],
            [-1,  1, -1,  1, -1,  1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1,  1,  1,  1],
            [-1,  1,  1, -1,  1, -1],
            [-1, -1,  1,  1, -1, -1],
            [-1,  1, -1,  1, -1,  1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1,  1,  1,  1],
            [-1,  1,  1, -1,  1,  1],
            [-1, -1,  1,  1, -1,  1],
            [-1,  1, -1,  1, -1, -1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1,  1,  1, -1],
            [-1,  1,  1, -1,  1,  1],
            [-1, -1,  1,  1, -1,  1],
            [-1,  1, -1,  1, -1,  1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1,  1,  1,  1],
            [-1,  1,  1, -1,  1, -1],
            [-1, -1,  1,  1, -1,  1],
            [-1,  1, -1,  1, -1, -1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1,  1,  1,  1],
            [-1,  1,  1, -1,  1,  1],
            [-1, -1,  1,  1, -1, -1],
            [-1,  1, -1,  1, -1,  1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1,  1,  1,  1, -1],
            [-1,  1, -1, -1,  1, -1],
            [-1,  1, -1,  1, -1,  1],
            [-1,  1,  1, -1, -1,  1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1,  1,  1,  1, -1],
            [-1,  1, -1, -1,  1,  1],
            [-1,  1, -1,  1, -1, -1],
            [-1,  1,  1, -1, -1,  1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1,  1,  1,  1,  1],
            [-1,  1, -1, -1,  1,  1],
            [-1,  1, -1,  1, -1,  1],
            [-1,  1,  1, -1, -1, -1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1,  1,  1,  1, -1],
            [-1,  1, -1, -1,  1,  1],
            [-1,  1, -1,  1, -1,  1],
            [-1,  1,  1, -1, -1, -1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1,  1,  1,  1,  1],
            [-1,  1, -1, -1,  1, -1],
            [-1,  1, -1,  1, -1,  1],
            [-1,  1,  1, -1, -1,  1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1,  1,  1,  1,  1],
            [-1,  1, -1, -1,  1,  1],
            [-1,  1, -1,  1, -1, -1],
            [-1,  1,  1, -1, -1,  1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1,  1,  1,  1, -1],
            [-1,  1, -1,  1,  1, -1],
            [-1,  1,  1, -1,  1,  1],
            [-1,  1,  1,  1, -1,  1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1,  1,  1,  1, -1],
            [-1,  1, -1,  1,  1,  1],
            [-1,  1,  1, -1,  1, -1],
            [-1,  1,  1,  1, -1,  1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1,  1,  1,  1,  1],
            [-1,  1, -1,  1,  1, -1],
            [-1,  1,  1, -1,  1, -1],
            [-1,  1,  1,  1, -1,  1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1,  1,  1,  1, -1],
            [-1,  1, -1,  1,  1,  1],
            [-1,  1,  1, -1,  1,  1],
            [-1,  1,  1,  1, -1, -1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1,  1,  1,  1,  1],
            [-1,  1, -1,  1,  1,  1],
            [-1,  1,  1, -1,  1, -1],
            [-1,  1,  1,  1, -1, -1],
        ]),
        
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1,  1,  1,  1,  1],
            [-1,  1, -1,  1,  1, -1],
            [-1,  1,  1, -1,  1,  1],
            [-1,  1,  1,  1, -1, -1],
        ]),
    ] # }}}

    results_full = []
    for X in Xmats:
        print "Evaluating matrix: X =\n%s\n" %(X)
        results = {}
        for _ in range(reps_per_mtx):
            res = single_run([[X, 0]], verbose)
            if res not in results:
                results[res] = 0
            results[res] += 1
        # end for
        results_full.append("X = \n%s\nresults = %s\n" %(X, results))
    # end for

    for result in results_full:
        print result
# end def}}}
def n5_gen_trap(needed = 20, trapfile = './trap5.pickle'): # {{{
    '''
    Find trapped matrices for n=5.
    '''
    reps_per_mtx = 100
    trapped = []
    total_runs = 0

    Xmats = [
        # Top right.
        np.matrix([
            [-1,  -1,  -1,  -1,  -1],
            [-1,  -1,   1,   1,   1],
            [-1,   1,  -1,  -1,   1],
            [-1,   1,  -1,   1,  -1],
            [ 1,  -1,  -1,   1,   1],
        ]),
    ]

    while True:
        # Select a base matrix.
        X = Xmats[ np.random.randint(len(Xmats)) ]

        # Run with this matrix.
        for i in range(reps_per_mtx):
            ret = single_run([[X, 8]], trap = True)
            total_runs += 1
            if ret is not None:
                trapped.append(ret)
            if i % 10 == 9:
                print "%d run" %(i+1)
                select.select([], [], [], 0.2)
        # end for
        print "num gathered = %d" %(len(trapped))
        if len(trapped) >= needed:
            break
        select.select([], [], [], 0.5)
    # end while

    print "%d total runs, %d trapped\nsaving trapped to %s" %(total_runs,
            len(trapped), trapfile)
    with open(trapfile, 'w') as fp:
        pickler = pickle.Pickler(fp)
        pickler.dump([total_runs, trapped])
    print "Done"
# end }}}
def n5_retry_trap(trapfile = './trap5.pickle', runs_per_mtx = 100): # {{{
    ret = None
    with open(trapfile, 'r') as fp:
        unpickler = pickle.Unpickler(fp)
        total_runs, ret = unpickler.load()

    times_trapped = [[0, 0, 0] for _ in range(len(ret))]
    for i, item in enumerate(ret):
        X, A, Y, Ui, U, tU = item
        # TODO: Deal with whether badness is about A or about X.
        # Redrawing A is really simple: just create a new A and re-calc Y.
        # Redrawing X may be best done by getting new extra cols.
        # Maybe get stats for redo, new X, new A.

        for _ in range(runs_per_mtx):
            res = repeat_run(X, A, Y)
            if res == 'SOLVED':
                pass
            elif res == 'TRAPPED':
                times_trapped[i][0] += 1
            else:
                raise RuntimeError, "Unexpected error: res = %s" %(res)
            # end if
        # end for

        # Redraw A, recalc Y.
        Anew = np.random.randn(A.shape[0], A.shape[1])
        Ynew = Anew * X
        for _ in range(runs_per_mtx):
            res = repeat_run(X, Anew, Ynew)
            if res == 'SOLVED':
                pass
            elif res == 'TRAPPED':
                times_trapped[i][1] += 1
            else:
                raise RuntimeError, "Unexpected error: res = %s" %(res)
            # end if
        # end for

        # Redraw X, recalc Y.
        Xnew = X.copy()
        ncol = X.shape[1] - X.shape[0]
        Xnew[:, -ncol:] = np.array(
                np.random.rand(X.shape[0], ncol) < 0.5, dtype = np.int
                ) * 2 - 1
        Ynew = Anew * X
        for _ in range(runs_per_mtx):
            res = repeat_run(Xnew, A, Ynew)
            if res == 'SOLVED':
                pass
            elif res == 'TRAPPED':
                times_trapped[i][2] += 1
            else:
                raise RuntimeError, "Unexpected error: res = %s" %(res)
            # end if
        # end for

        print "Done with item %d" %(i)
        select.select([], [], [], 0.5)
    # end for

    for i, numbers in enumerate(times_trapped):
        print "%2d: %s" %(i, numbers)
    print "total: %d / %d" %(sum([t[0] for t in times_trapped]), 
            runs_per_mtx * len(ret))
    print "original = %d / %d" %(len(ret), total_runs)
# end}}}
def trap_by_X_k(Xmats, klim, reps_per_k = 100, plotfile = None): # {{{
    klow = max([X.shape[1] for X in Xmats])
    trapped = [[0, 0] for i in range(klim + 1)]
    n = Xmats[0].shape[0]

    for k in range(klow, klim + 1):
        for _ in range(reps_per_k):
            X = Xmats[ np.random.randint(len(Xmats)) ]
            ret = single_run([[X, k - X.shape[1]]])
            if ret == 'ERROR':
                trapped[k][1] += 1
            if ret == 'TRAPPED':
                trapped[k][0] += 1
        # end for
    # end for

    print "TRAPPED"
    for k in range(klow, klim + 1):
        print "%2d: %3d" %(k, trapped[k][0])
    print "ERRORS"
    for k in range(klow, klim + 1):
        print "%2d: %3d" %(k, trapped[k][1])

    if plotfile is not None:
        trapval = [sum(trapped[k]) / float(reps_per_k) for k in range(len(trapped))]
        plt.plot(range(klow, klim+1), trapval[klow: ])
        plt.axis([klow, klim, 0, max(trapval)])
        plt.xlabel("k")
        plt.ylabel("trap probability")
        plt.title("Trapped vs. k (n = %d) [%d runs per k]" %(n, reps_per_k))
        plt.savefig(plotfile, format='png')
        plt.show()
# end}}}
def trap_experiment(): #{{{
    # Base X matrix to use.
    X4 = [
        np.matrix([
            [1,  1,  1,  1,  1],
            [1,  1, -1, -1,  1],
            [1, -1,  1, -1,  1],
            [1, -1, -1,  1, -1],
        ]),
    ]

    X5 = [
        np.matrix([
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1,  1,  1, -1],
            [-1, -1,  1, -1,  1,  1],
            [-1,  1, -1, -1,  1,  1],
            [-1,  1,  1,  1, -1, -1],
        ]),
    ]

    #trap_by_X_k(X4, 16, reps_per_k = 1000, plotfile = "./n4_trapped_k.png")
    trap_by_X_k(X5, 20, reps_per_k = 1000, plotfile = "./n5_trapped_k.png")
# end}}}

def second_trap_experiment(reps_per = 100):
    '''Parse all the given trap cases and see how frequently they result in
    traps
    '''
    print 'Loading cases...'
    cases = parse_trap.load_cases('trap_exp.txt')
    print 'Found %d cases.' % len(cases)

    results = []

    for case in cases:
        rc = { 'trapped' : 0, 'solved' : 0, 'runout' : 0, 'error' : 0 }

        if not case.is_valid():
            continue

        sys.stdout.write('Starting case...')
        sys.stdout.flush()
        X = np.round(case.Ainv * case.Y)
        A = np.linalg.inv( case.Ainv )

        for _ in range(reps_per):
            res = repeat_run( X, A, case.Y )
            if res == 'SOLVED':
                print 'SOLVED'
                rc['solved'] += 1
            elif res == 'RUNOUT':
                print 'RUNOUT'
                rc['runout'] += 1
                if rc['runout'] > 9:
                    break
            elif res == 'TRAPPED':
                print 'TRAPPED'
                rc['trapped'] += 1
            else:
                print "Unexpected error: res = %s" %(res)
                rc['error'] += 1

        print rc
        results.append(rc)
        
    for r in results:
        print r

def n6(verbose = False): # {{{
    '''
    n=6 single run
    '''
    Xmats = [
        [ np.matrix([
            [-1,  1,  1,  1,  1,  1],
            [ 1, -1,  1,  1,  1,  1],
            [ 1,  1, -1,  1,  1,  1],
            [-1, -1, -1, -1,  1,  1],
            [-1, -1, -1,  1, -1,  1],
            [-1, -1, -1,  1,  1, -1],
        ]), 8],
    ]

    return single_run(Xmats, verbose)
# end def}}}
def n8(verbose = False): # {{{
    '''
    n=8 single run
    '''
    Xmats = [
        [ np.matrix([
            [  1,  1,  1,  1,  1,  1,  1,  1],
            [  1,  1, -1, -1,  1,  1, -1, -1],
            [  1, -1, -1,  1,  1, -1, -1,  1],
            [  1, -1,  1, -1,  1, -1,  1, -1],
            [  1,  1,  1,  1, -1, -1, -1, -1],
            [  1,  1, -1, -1, -1, -1,  1,  1],
            [  1, -1, -1,  1, -1,  1,  1, -1],
            [  1, -1,  1, -1, -1,  1, -1,  1],
        ]), 8],
        [ np.matrix([
            [  1,  1,  1, -1,  1,  1,  1, -1],
            [  1,  1, -1,  1,  1,  1, -1,  1],
            [  1, -1,  1,  1,  1, -1,  1,  1],
            [ -1,  1,  1,  1, -1,  1,  1,  1],
            [  1,  1,  1, -1, -1, -1, -1,  1],
            [  1,  1, -1,  1, -1, -1,  1, -1],
            [  1, -1,  1,  1, -1,  1, -1, -1],
            [ -1,  1,  1,  1,  1, -1, -1, -1],
        ]), 8],
    ]

    return single_run(Xmats, verbose)
# end def}}}

def read_error_UY(fname): # {{{
    '''
    Read a pickled [U, Y] contained in file +fname+, create a tableau.
    Then run checks on the specific current state U.
    '''
    with open(fname, "r") as fp:
        unpickler = pickle.Unpickler(fp)
        U, Y = unpickler.load()

    print "U = \n%s\nY = \n%s\n" %(U, Y)
    t = FlexTab(U, Y, verbose = True)
    t.to_simplex_form()
    grad = t.calc_grad().flatten()
    
    t.snapshot()
    for v in range(len(grad)):
        effect = t.flip_gradient(v, grad)
        print "v = %d, effect = %f" %(v, effect)
        if effect < -t.zthresh:
            print "Negative gradient: skip\n"
        else:
            t.flip(v)
            if not t.eval_vertex():
                print "Invalid vertex\n"
        # end if
        t.restore()
    # end for
    t.restore(pop = True)
    print "Trapped? %s" %(t.is_trapped())

# end def}}}
# end function set}}}

# multi trial functions{{{
def tester_clean(reps = 100, verbose = False): # {{{
    '''
    Generalized tester for matrices where for each run it is ensured that the
    initial BFS finder returns no bad columns.
    This is generally intended to be run with matrices for which it is
    guaranteed that an ATM will be recovered, but this requirement is not
    necessary.
    '''
    # Setup matrices to test.
    tests = [
        np.matrix([
            [1,  1],
            [1, -1],
        ]),
        
        np.matrix([
            [1,  1,  1],
            [1, -1, -1],
        ]),

        np.matrix([
            [1,  1,  1, -1, -1],
            [1, -1, -1,  1, -1],
        ]),
        
        np.matrix([
            [ 1,  1,  1,  1],
            [ 1,  1, -1, -1],
            [ 1, -1,  1, -1],
        ]),

        np.matrix([
            [ 1,  1,  1,  1, -1],
            [ 1,  1, -1, -1,  1],
            [ 1, -1,  1, -1, -1],
        ]),

        np.matrix([
            [ 1,  1,  1,  1, -1, -1, -1, -1],
            [ 1,  1, -1, -1,  1,  1, -1, -1],
            [ 1, -1,  1, -1,  1, -1,  1, -1],
        ]),

        np.matrix([
            [1,  1,  1,  1,  1],
            [1,  1, -1, -1,  1],
            [1, -1,  1, -1,  1],
            [1, -1, -1,  1, -1],
        ]),

        np.matrix([
            [1,  1,  1,  1,  1, -1],
            [1,  1, -1, -1,  1,  1],
            [1, -1,  1, -1,  1, -1],
            [1, -1, -1,  1, -1, -1],
        ]),
    ]

    successes = [0 for _ in range(len(tests))]
    for i, mtx in enumerate(tests):
        for _ in range(reps):
            if flex_run(mtx, 0, verbose):
                successes[i] += 1
    
    for i, mtx in enumerate(tests):
        print "X = \n%s\n%d / %d" %(mtx, successes[i], reps)

# end def}}}
def tester_badcols(reps = 100, verbose = False): # {{{
    '''
    Function for testing matrices where, for each matrix, we specify a given
    number of +bad+ columns that the initial BFS finder must return [the BFS
    finder is run repeatedly until this specified number of bad columns occurs].
    '''
    tests = [
        [
            np.matrix([
                [1,  1,  1,  1,  1],
                [1,  1, -1, -1,  1],
                [1, -1,  1, -1,  1],
                [1, -1, -1,  1, -1],
            ]), 1
        ],
        [
            np.matrix([
                [1,  1,  1,  1,  1, -1],
                [1,  1, -1, -1,  1,  1],
                [1, -1,  1, -1,  1, -1],
                [1, -1, -1,  1, -1, -1],
            ]), 2
        ],
        [
            np.matrix([
                [1,  1,  1,  1,  1, -1,  1],
                [1,  1, -1, -1,  1, -1, -1],
                [1, -1,  1, -1,  1, -1, -1],
                [1, -1, -1,  1, -1, -1, -1],
            ]), 2
        ],
        [
            np.matrix([
               [-1,  -1,  -1,  -1,  -1,  1],
               [-1,  -1,  -1,   1,   1,  1],
               [-1,  -1,   1,  -1,   1,  1],
               [-1,   1,  -1,  -1,   1, -1],
               [-1,   1,   1,   1,  -1,  1],
            ]), 1
        ],
        [
            np.matrix([
               [-1,  -1,  -1,  -1,  -1,  1,  1],
               [-1,  -1,  -1,   1,   1,  1, -1],
               [-1,  -1,   1,  -1,   1,  1,  1],
               [-1,   1,  -1,  -1,   1, -1, -1],
               [-1,   1,   1,   1,  -1,  1,  1],
            ]), 2
        ],
    ]

    targets = [i for i in range(len(tests))]
    successes = [0 for _ in range(len(tests))]
    for i in targets:
        mtx, bad_cols = tests[i]
        for _ in range(reps):
            if flex_run(mtx, bad_cols, verbose):
                successes[i] += 1
    
    for i in targets:
        print "X = \n%s\n%d / %d" %(tests[i][0], successes[i], reps)
# end def}}}
def many_func(func, reps = 100, verbose = False, stop_on_error = False): # {{{
    results = {}
    total = 0
    resfmt = lambda: "[%d]: %s" %(total,
            ', '.join(map(lambda k: '%s => %d' %(k, results[k]),
                    results.keys())))

    for _ in range(reps):
        res = func(verbose = verbose)
        if not res in results:
            results[res] = 1
        else:
            results[res] += 1
        # end if
        total += 1
        if total % 10 == 0:
            print resfmt()
            select.select([], [], [], 2.0)
        if res == 'ERROR' and stop_on_error:
            break
    # end while

    print resfmt()
# end def}}}

def run_many(func, reps, verbose = False, stop = False): # {{{
    '''
    Generalized function for running mulitple instances of any given function
    +func+ for +reps+ repetions.  Note that +func+ should return a bool.
    '''
    success = 0
    for r in range(reps):
        if func(verbose):
            success += 1
        elif stop:
            break
    print "%d / %d = %f" %(success, reps, float(success) / reps)
# end def}}}
# end function set}}}

# bad column anomaly functions{{{
def bad_counter(): # {{{
    A_h1 = [ np.matrix([
            [1,  1,  1,  1,  1],
            [1,  1, -1, -1,  1],
            [1, -1,  1, -1,  1],
            [1, -1, -1,  1, -1],
        ]), 0]

    A_h1_ok = [ np.matrix([
            [1,  1,  1,  1,  1, -1],
            [1,  1, -1, -1,  1, -1],
            [1, -1,  1, -1,  1,  1],
            [1, -1, -1,  1, -1, -1],
        ]), 0]

    A_h1_rand = [ np.matrix([
            [1,  1,  1,  1,  1],
            [1,  1, -1, -1,  1],
            [1, -1,  1, -1,  1],
            [1, -1, -1,  1, -1],
        ]), 1]

    A_5_0 = [ np.matrix([
            [-1,  -1,  -1,  -1,  -1,  1],
            [-1,  -1,  -1,   1,   1,  1],
            [-1,  -1,   1,  -1,   1,  1],
            [-1,   1,  -1,  -1,   1, -1],
            [-1,   1,   1,   1,  -1,  1],
    ]), 0]

    A_5_0_p1 = [ np.matrix([
            [-1,  -1,  -1,  -1,  -1,  1],
            [-1,  -1,  -1,   1,   1,  1],
            [-1,  -1,   1,  -1,   1,  1],
            [-1,   1,  -1,  -1,   1, -1],
            [-1,   1,   1,   1,  -1,  1],
    ]), 1]

    A_5_0_p5 = [ np.matrix([
            [-1,  -1,  -1,  -1,  -1],
            [-1,  -1,  -1,   1,   1],
            [-1,  -1,   1,  -1,   1],
            [-1,   1,  -1,  -1,   1],
            [-1,   1,   1,   1,  -1],
    ]), 5]

    # Bottom right
    A_5_1_p5 = [ np.matrix([
            [-1,  -1,  -1,  -1,  -1],
            [-1,  -1,  -1,   1,   1],
            [-1,   1,   1,  -1,   1],
            [ 1,  -1,   1,  -1,   1],
            [ 1,   1,  -1,  -1,   1],
    ]), 5]
    
    # Bottom left
    A_5_2_p2 = [ np.matrix([
            [-1,  -1,  -1,  -1,  -1],
            [-1,  -1,  -1,   1,   1],
            [-1,   1,   1,  -1,   1],
            [-1,  -1,   1,   1,  -1],
            [ 1,  -1,   1,  -1,   1],
    ]), 2]

    # Top right
    A_5_3_p2 = [ np.matrix([
            [-1,  -1,  -1,  -1,  -1],
            [-1,  -1,   1,   1,   1],
            [-1,   1,  -1,  -1,   1],
            [-1,   1,  -1,   1,  -1],
            [ 1,  -1,  -1,   1,   1],
    ]), 2]

    A_5_3_p8 = [ np.matrix([
            [-1,  -1,  -1,  -1,  -1],
            [-1,  -1,   1,   1,   1],
            [-1,   1,  -1,  -1,   1],
            [-1,   1,  -1,   1,  -1],
            [ 1,  -1,  -1,   1,   1],
    ]), 8]

    tests = [A_5_2_p2]
    rand_mtx_per_base = 5
    reps_per_mtx = 10

    for Xbase, ncol in tests:
        X = np.concatenate((Xbase, np.zeros([Xbase.shape[0], ncol])), axis=1)
        for _ in range(rand_mtx_per_base):
            # Create random extra cols for this base mtx Xbase.
            X[:, -ncol:] = np.array(
                    np.random.rand(X.shape[0], ncol) < 0.5, dtype = np.int
                    ) * 2 - 1
            # Run some reps with this specific rand mtx.
            good_bad_cols(X, reps_per_mtx)
        # end for
    # end for

# end def}}}

def good_bad_cols(X, reps, save_bad_file = None): # {{{
    UY = None
    U = None
    Y = None
    A = None
    num_cols = X.shape[1]
    counts = [0 for i in range(num_cols + 1)]
    bad_instances = []

    for _ in range(reps):
        UY, U, Y, A, Ui = trial(X)
        t = FlexTab(U, Y, verbose = False)
        num_bad_cols = X.shape[1] - t.num_good_cols()
        counts[num_bad_cols] += 1

        # Save instances where num_bad_cols > 1.
        if num_bad_cols > 1:
            instance = { 'U':U, 'Ui':Ui, 'Y':Y, 'UY':UY, 'A':A, 'X':X.copy() }
            bad_instances.append(instance)
    # end for

    # Save bad instances
    if save_bad_file is not None and len(bad_instances) > 0:
        with open(save_bad_file, 'w') as fp:
            pickler = pickle.Pickler(fp)
            examine_instances(bad_instances)
            pickler.dump(bad_instances)
        # end with
    # end if

    # Show +good+ cols in output.
    counts.reverse()
    print "X = \n%s\ncounts:\n%s\n%s\n%s\n" %(
            X,
            '|'.join(map(lambda i: "%4d" %(i), range(len(counts)))),
            '-' * 80,
            '|'.join(map(lambda c: "%4d" %(c), counts))
            )
    n = X.shape[0]
    print "num with at least %d good = %d\n" %(n,
            reduce(lambda a, e: a + e, counts[n:], 0))
# end def}}}
def examine_instances(instances): # {{{
    for i, instance in enumerate(instances):
        print "\n***BEGIN INSTANCE %d***\n%s\n" %(i, '-' * 40)
        print "X = \n%s\n" %(instance['X'])
        print "A = \n%s\n" %(instance['A'])
        print "Ui = \n%s\n" %(instance['Ui'])
        print "U = \n%s\n" %(instance['U'])
        print "Y = \n%s\n" %(instance['Y'])
        print "UY = \n%s\n" %(instance['UY'])
        print "\n***END INSTANCE %d***\n%s\n" %(i, '-' * 40)
    # end for
# end def}}}
def load_bad_instances(bad_instance_file): # {{{
    with open(bad_instance_file, 'r') as fp:
        unpickler = pickle.Unpickler(fp)
        bad_instances = unpickler.load()
        examine_instances(bad_instances)
    # end with
# end def}}}
# end function set}}}

# Tableau classes{{{
class FlexTab(object): # {{{
    def __init__(self, U, Y, verbose = False, zthresh = 1e-9): # {{{
        # Setup administrative variables / functions.
        self.set_done_funcs()
        self.verbose = verbose
        self.zthresh = zthresh

        # Setup columns, including determining initial good columns.
        # Potential splice point for dealing with initial BFS.
        self.set_columns(U, Y)

        # Rows in which a given U variable is basic.
        self.urows = [None for i in range(self.n**2)]

        # Mapping of an index [0, self.n**2] to an actual slackvar pair that we
        # will use in flipping a value in UY.  These must stem from linearly
        # independent cols of Y.  Tableau formation does this automatically.
        self.vmap = [None for i in range(self.n**2)]

        # 2-D array giving, for each of the n constraint sets, restrictions
        # on the overall UY.
        self.extra_constr = [ [] for i in range(self.n)]

        # Prellocate space for the gradient and the current U est A^-1 matrix.
        self.grad = np.zeros((self.n, self.n))

        # Allocate space for self.U matrix, which holds the estimate of A^-1.
        # This is not updated every time self.X changes, but only when needed.
        self.U = np.zeros((self.n, self.n))

        # Stack of states used for backtracking.
        self.states = []
        # Set of visited vertices (bfs's).
        self.visited = set()

        # We need 2nk rows: 2 per each unrolled value in |Uy|_\infty \leq 1.
        # We need 2n^2 + 2nk + 1 columns:
        #   Each of the n^2 variables needs 2 vars so we can form x'_0 - x'_1.
        #   Each of the 2nk constraints needs a slack variable.
        #   Last column is for the right hand side.
        self.rows = np.zeros([2 * self.nk, 2 * self.n**2 + 2 * self.nk + 1])

        # The 2nk + 2n^2 X variables are as just described above.
        self.X = np.zeros(2 * self.n**2 + 2 * self.nk)

        # We are given values for the n^2 X variables, but these are real
        # numbers, so we convert x_0 = x'_0 - x'_1.
        for i, elt in enumerate(np.array(U).flatten()):
            if elt >= 0:
                self.X[2 * i] = elt
                self.X[2 * i + 1] = 0
            else:
                self.X[2 * i] = 0
                self.X[2 * i + 1] = -elt

        # Add all constraints.
        self.cur_row = 0
        self.set_constraints()
    # end def}}}
    def set_columns(self, U, Y): # {{{
        '''
        Set all columns of Y into self.  Every column such that UY = \pm 1 is a
        +good+ column.

        Set self.Y to all of the +good+ columns of Y.  Note self.Y will be used
        directly in vertex-hopping.

        Set setf.Yfull to the full Y, regardless of whether all cols are +good+.

        If there are any +bad+ columns, then set self.reduced to True.  In this
        case, after the vertex-hopping functionality finds -any- valid vertex,
        meaning UYfull = \pm 1, the algorithm will stop, rebuild a new tableau
        with this newly found U [that is a true BFS], and then vertex hop again
        [since this new U gives no bad columns, this second tableau will be the
        simpler case].
        '''
        # Determine U * Y, set self.good_cols accordingly.
        self.prod = U.dot(Y)
        self.good_cols = []
        for j in range(self.prod.shape[1]):
            # If every entry is within zthresh of \pm 1, then col is good.
            if np.sum( np.abs(np.abs(self.prod[:,j]) - 1) > 1e-9 ) == 0:
                self.good_cols.append(j)

        # Determine if all columns are good, setup accordingly.
        self.n = U.shape[0]
        self.k = len(self.good_cols)
        self.nk = self.n * self.k
        self.Y = np.zeros((self.n, self.k))
        for j, col in enumerate(self.good_cols):
            self.Y[:, j] = Y[:, col].flatten()

        if len(self.good_cols) == Y.shape[1]:
            self.Yfull = self.Y
            self.reduced = False
        else:
            self.Yfull = Y
            self.reduced = True
        # end if
    # end def}}}
    def set_constraints(self): # {{{
        '''
        Set the constraints that will be used for vertex hopping.  These are
        only the constraints imposed by self.Y, thus only the +good+ columns of
        Y.  The other constraints are captured in self.extra_constr.
        '''
        for i in range(self.n):
            # Foreach row, add the constraints from the _used_ columns.
            for j in range(self.k):
                # The i^th row uses variables in range: i * n .. i * (n +1).
                # We are constraining |Uy|_\infty \leq 1.
                # constraints are an n-vector of tuples [(varnum, coeff), ... ].
                constr = [(i * self.n + k, self.Y[k, j]) for k in range(self.n)]
                self.add_constraints(constr)
            # end for
        # end for
    # end def}}}
    def add_constraints(self, constr): # {{{
        '''
        Add 2 new constraints based on the content of _constr_, which should be
        an array of 2-tuples of the form (varnum: int, coeff: float).
        '''
        # Setup the two new constraints based on vars / coeffs given.
        for var, coeff in constr:
            # Example: 6x_0 will become:
            #   self.cur_row + 0:  6x'_0 - 6x'_1 <= RHS = 1
            #   self.cur_row + 1: -6x'_0 + 6x'_1 <= RHS = 1
            self.rows[self.cur_row, var * 2] = coeff
            self.rows[self.cur_row, var * 2 + 1] = -coeff
            self.rows[self.cur_row + 1, var * 2] = -coeff
            self.rows[self.cur_row + 1, var * 2 + 1] = coeff

        # Need to determine values of the slack variables for this pair of
        # constraints.  Compute the value of the LHS of the constraint.  One of
        # the pair will be tight, so slack will be 0; other slack will be 2.
        # Compute: val = \sum_{2n^2 LHS vars in this constr_set} a_{ij} * x_j
        base = self.cur_row / (2 * self.k)
        val = reduce(lambda a, j:
                a + self.rows[self.cur_row, j] * self.X[j],
                range(2 * base * self.n, 2 * (base + 1) * self.n), 0)

        # Slack var coeffs both 1.0.
        self.rows[self.cur_row, 2 * self.n**2 + self.cur_row] = 1.0
        self.rows[self.cur_row + 1, 2 * self.n**2 + self.cur_row + 1] = 1.0

        # RHS values are both 1.0.
        self.rows[self.cur_row, -1] = 1.0
        self.rows[self.cur_row + 1, -1] = 1.0

        # Slack var values must be set so LHS = RHS = 1.0.
        self.X[self.cur_row + 2 * self.n**2] = 1.0 - val
        self.X[self.cur_row + 2 * self.n**2 + 1] = 1.0 + val

        # This completes the addition of two constraints (rows).
        self.cur_row += 2
    #end # }}}
    def num_good_cols(self): # {{{
        return len(self.good_cols)
    # end def}}}
    def num_indep_cols(self): # {{{
        # Determine rank of Y.
        u, s, v_t = scipy.linalg.svd(self.Y)
        if self.verbose:
            print "SVD(Y):\nu = \n%s\ns = %s\nv_t = %s\n" %(u, s, v_t)
        return np.sum(s > 1e-6)
    # end}}}
    def __str__(self): # {{{
        '''
        Return a verbose string representation of the current tableau.
        '''
        s = ''

        # X-values: print 4 values per line.
        xv = []
        for i in range(len(self.X)):
            xv.append('x_%d = %f' %(i, self.X[i]))
            if i % 4 == 3:
                s += ' '.join(xv) + "\n"
                xv = []
        s += "\n"

        # Constraints: one per line.
        for i, r in enumerate(self.rows):
            line = []
            for j, v in enumerate(r[:-1]):
                if np.abs(self.rows[i, j]) > 1e-9:
                    line.append('%.2fx_%d' %(self.rows[i, j], j))
            line = ' + '.join(line)
            s += '%s == %.2f\n' %(line, self.rows[i, -1])
        return s
    #end # }}}

    def snapshot(self): # {{{
        '''
        Take a snapshot of the current state.
        '''
        # TODO: Is this enough to save.  Certainly should save more for
        # efficiency, but is more needed immediately for correctness?
        snap = {
            'X' : self.X.copy(),
            'vmap' : [v for v in self.vmap],
            'rows' : self.rows.copy(),
            'U' : self.U.copy(),
        }
        self.states.append(snap)
    #end # }}}
    def restore(self, pop = False): # {{{
        '''
        Restore the state snapshot at the top of the stack.
        INPUT:  pop = if True, pop the state off the stack; else just copy it.
        OUTPUT: True if state restored; False if state stack was empty.
        '''
        if len(self.states) == 0:
            return False

        if pop:
            snap = self.states.pop()
            self.X = snap['X']
            self.rows = snap['rows']
            self.U = snap['U']
            self.vmap = snap['vmap']
        else:
            snap = self.states[-1]
            self.X = snap['X'].copy()
            self.rows = snap['rows'].copy()
            self.U = snap['U'].copy()
            self.vmap = [v for v in snap['vmap']]

        return True
    # end }}}
    def vertex(self): # {{{
        '''
        Return the repesentation, in 0-1 form, at the current vertex X.  This is
        determined by examining which of the slack X-variables are tight.
        '''
        return ''.join(map(lambda v: '0' if (v & 0x1) == 0 else '1', self.vmap))
    # end}}}
    def mark_visited(self): # {{{
        # XXX:
        if self.verbose:
            print "vertex = %s" %(self.vertex())
        self.visited.add(self.vertex())
    # end def}}}
    def is_flip_visited(self, v): # {{{
        newstr = self.vertex()
        newstr = [newstr[i] for i in range(len(newstr))]
        newstr[v] = '1' if newstr[v] == '0' else '0'
        newstr = ''.join(newstr)
        return newstr in self.visited
    # end def}}}

    def solve(self): # {{{
        if not self.to_simplex_form():
            print "Unable to convert orig tab to simplex form:\n%s\n" %(self)
            return False
        if self.verbose:
            print "%s" %(self)
            print "%s" %(self.extra_constr)
        success = self.hop()
        if not success:
            if self.reduced:
                print "Unable to find initial BFS"
            else:
                print "From initial BFS unable to find optimum"
            return False
        # end if

        if self.reduced:
            t = FlexTab(self.U, self.Yfull, verbose = self.verbose)
            if not t.solve():
                print "Reduced mtx: Unable to solve expanded self"
                return False
            self.U = t.U
        # end if

        return True
    # end def}}}

    def to_simplex_form(self): # {{{
        '''
        Create simplex-style tableau, which will be stored in self.rows.  Note
        that self.X is of length 2 * self.nk.  There are 2 * self.nk
        overall constraints in |UY|_\infty \leq 1.
        '''
        # Zero out any entries in X that are solely float imprecision.
        self.X[np.abs(self.X) < self.zthresh] = 0

        # There are n groups, each of k equation pairs, and within a group every
        # equation pair uses the same set of X variables.
        for constr_set in range(self.n):
            # Grab each [of the k] equation pairs individually.  Idea is to
            # iterate over all still-available basic vars, then if one exists
            # such that self.rows[row, var] > zthresh, then use var as the basic
            # var for this eq_set [where row = 2 * eq_set].
            avail_basic = set([v for v in range(constr_set * self.n,
                        (constr_set + 1) * self.n)])
            for j in range(self.k):
                # eq_pair is the pair of equations we are currently focused on.
                eq_pair = constr_set * self.k + j

                # Exactly one of pairs should have a zero slack var value, since
                # exactly one constraint is tight.
                zeroslack = [q for q in range(eq_pair * 2, eq_pair * 2 + 2) \
                        if np.abs(self.X[2 * self.n**2 + q]) < self.zthresh]
                if len(zeroslack) != 1:
                    if self.verbose:
                        print "num zero slackvars in pair != 1: eq_set %d" %(j)
                    return False

                # If there is an available var with a nonzero coeff, use it.
                if self.verbose:
                    print "avail_basic = %s" %(avail_basic)
                useable = [v for v in avail_basic if \
                    abs(self.rows[2 * eq_pair, 2 * v]) > self.zthresh]
                if self.verbose:
                    print "constr = %d, j = %d eq_pair = %d, zs = %s, u = %s" %(
                            constr_set, j, eq_pair, zeroslack, useable)

                if len(useable) > 0:
                    # Use the row where slack = 0 to turn an original U-var into
                    # a basic var.  For ease we use the original x_j, so we make
                    # a basic from whichever of x_{2j}, x_{2j+1} is nonzero.
                    uvar = useable[0] # constr_set * self.n + j
                    avail_basic.remove(uvar)
                    if not self.set_basic(uvar, zeroslack[0]):
                        return False
                else:
                    tgtrow = zeroslack[0]
                    srcrow = tgtrow + 1 if not tgtrow & 0x1 else tgtrow - 1
                    self.rows[tgtrow] += self.rows[srcrow]

                if self.verbose:
                    print "%s\n" %self
            # end for
        # end for

        self.tableau_mappings()
        return True
    # end}}}
    def set_basic(self, j, pivot_row): # {{{
        '''
        Create a simplex-style basic variable from the original x_j.  Note that
        the original x_j \in R is now divided into the difference of variables
        \in R^+: x_{2j} and x_{2j+1}.  We make a basic variable from whichever
        of x_{2j}, x_{2j+1} is nonzero.  The selected variable has its
        coefficient in pivot_row set to 1, and zeroed everywhere else.

        INPUT:  j = original x-var: the new basic var will be whichever of
                    x_{2j}, x_{2j+1} is nonzero.
                pivot_row = row in which the selected new basic var should have
                    coefficient set to 1.
        OUTPUT: True on success.
        '''
        # Basic var = whichever of x_{2j}, x_{2j+1} is nonzero.
        nonzero = [q for q in range(2*j, 2*j + 2) \
                if np.abs(self.X[q]) > self.zthresh]
        # Exactly one var should be nonzero, since x_{2j} - x_{2j+1} is the
        # value of the original u_j.
        if len(nonzero) != 1:
            print "pair of xvars should have exactly 1 nonzero: %s" %(nonzero)
            return False
        tgtvar = nonzero[0]

        # Make coeff of pivot_row[tgtvar] = 1.
        self.rows[pivot_row] /= self.rows[pivot_row, tgtvar]
        
        # Every other row gets tgtvar eliminated.
        s = j / self.n * self.k
        for i in range(2*s, 2*(s + self.k)):
            if i == pivot_row: continue
            self.rows[i] -= self.rows[i, tgtvar] * self.rows[pivot_row]

        return True
    # end}}}
    def tableau_mappings(self): # {{{
        # Evaluate each constr_set [which corresponds to 1 row of U] separately.
        for constr_set in range(self.n):
            vvars = set()
            for row in range(2 * self.k * constr_set, 2 * self.k * (constr_set + 1)):
                uvars, slackvars = self.categorize_row(row)
                if self.verbose:
                    print "row  = %d, uvars = %s, slackvars = %s" %(
                            row, uvars, slackvars)
    
                # If there are any uvars, there must be 2, both from same X-var.
                if len(uvars) > 0:
                    if len(uvars) != 2 or uvars[0] >> 1 != uvars[1] >> 1:
                        # This fails if not enough lin indep cols.
                        raise ValueError, "Invalid: row = %d, uvars = %s" %(
                                row, uvars)
                    # Get the original U-var, which must be currently unmapped.
                    uvar = uvars[0] >> 1
                    if self.urows[uvar] is not None:
                        raise ValueError, "urows[%d] redef in row %d" %(uvar, row)
                    # Of the 2 coeffs for orig U-var: one is +1 and other -1.
                    if self.rows[row, 2*uvar] > 0:
                        xplus, xminus = 2*uvar, 2*uvar + 1
                    else:
                        xplus, xminus = 2*uvar + 1, 2*uvar
                    self.urows[uvar] = [row, xplus, xminus]

                    # All of the vars in RHS are "normal" slackvars, meaning these
                    # will be actually flipped.  Normal slackvars may be exactly
                    # those that are slacks in equations with 2 LHS vars?
                    vvars.update(set(slackvars))

                else:
                    # If there are exactly 2 slackvars and both correspond to
                    # the same +base+ equation, then this is dealt w in vvars.
                    eqs = [(q - 2 * self.n**2) >> 1 for q in slackvars]
                    if len(slackvars) != 2 or eqs[0] != eqs[1]:
                        self.add_extra_constr(constr_set, row, slackvars)
                # end if-else
            # end for row

            vvars = sorted([i for i in vvars])
            self.vmap[constr_set * self.n : (constr_set + 1) * self.n] = vvars
            if self.verbose:
                print "constr_set = %d: vvars = %s, extra_constr = %s" %(
                        constr_set, vvars, self.extra_constr[constr_set])
        # end for constr_set
        if self.verbose: # XXX:
            print "urows = %s" %(self.urows)
            print "vmap = %s" %(self.vmap)
            print "extra_constr = %s" %(self.extra_constr)
    # end def}}}
    def categorize_row(self, row): # {{{
        # Return arrays of variables that are:
        #   1. uvars: X variables derived from U variables.LHS (Potential basic)
        #   2. slackvars: X variables that are slack variables

        # Determine which series we are in, where the series is the row of U
        # from which these constraints are derived.
        g = row / (2 * self.k)

        # The 2n potential U-variables are those from the relevant row.
        uvars = [x for x in range(2 * self.n * g, 2 * self.n * (g+1)) \
            if abs(self.rows[row, x]) > self.zthresh
        ]

        # The 2k potential slack vars are those involving these same U rows.
        slackvars = [x for x in range(2 * self.n**2 + 2 * self.k * g, \
                2 * self.n**2 + 2 * self.k * (g+1)) \
            if abs(self.rows[row, x]) > self.zthresh
        ]

        #print "row = %d, uvars = %s, slackvars = %s" %(row, uvars, slackvars)
        return uvars, slackvars
    # end def}}}
    def add_extra_constr(self, constr_set, row, slackvars): # {{{
        # Expect that every RHS will equal 2.
        if abs(self.rows[row, -1] - 2.00) > 1e-9:
            raise ValueError, "Unexpected RHS"

        # Calculate which constraint set this belongs to, and the set's base.
        #constr_set = row / (2 * self.k)
        
        # Calculate +base+ since +v+ will be interpreted as offset from base.
        base = 2 * self.n**2 + constr_set * (2 * self.k)
        new_constr = []
        for v in slackvars:
            coeff = self.rows[row, v]
            coeff = np.round(coeff, 5) # TODO: Issue if irrantional, possible?
            prodmul = -1 if v & 0x1 == 0 else 1
            uyvar = (v - base) >> 1
            new_constr.append((uyvar, prodmul, coeff))
        # end for

        self.extra_constr[constr_set].append(new_constr)
        if self.verbose:
            print "row = %d\nconstr = %s" %(row, self.extra_constr[constr_set])
    # end def}}}

    def hop(self): # {{{
        self.mark_visited()
        if self.verbose:
            print "initial obj = %.5f" %(self.calc_obj())

        # Principal vertex hopping loop.
        total_hops = 0
        while True:
            if self.verbose:
                print "hopcount = %d, visited = %d" %(total_hops, len(self.visited))
            keep = self.eval_vertex()
            if not keep:
                if self.verbose:
                    print "Bad vertex, attempting backtrack"
                if not self.restore(pop = True):
                    #raise ValueError, "Empty state stack"
                    if self.verbose:
                        print "Empty state stack"
                    return False
                else:
                    if self.verbose:
                        print "Successful restore"
                    # TODO: Should have neighbor search info in snapshot.

            #v_det = self.calc_v_det() # XXX: is this effective?

            if self.is_done():
                if self.verbose:
                    print "End obj = %.5f" %(self.calc_obj())
                return True
            if self.verbose:
                print "obj = %.5f" %(self.calc_obj())
            # XXX:
            #if abs(v_det) == 16:
            #    print "v_det = 16 but not done"

            flip_idx, effect, num_nonnegative_unvisited = self.search_neighbors()

            if flip_idx is None or effect < -self.zthresh:
                print "Nowhere nonnegative to go, attempting backtrack"
                if not self.restore(pop = True):
                    #raise ValueError, "Empty state stack"
                    print "Empty state stack"
                    return False
                continue

            # Flip specified index, mark vertex visited, take snapshot.
            # XXX: TODO: Ensure this doesn't break anything.
            #if num_nonnegative_unvisited > 1:
            self.snapshot()
            if self.verbose:
                print "Hop %s" %("better" if effect > self.zthresh else "equal")
            self.flip(flip_idx)
            total_hops += 1
            self.mark_visited()
            if total_hops > 1000:
                return False
        # end while
    # end def}}}
    def search_neighbors(self): # {{{
        '''
        Returns:
            best_idx = index of best_effect
            best_effect = best effect for flip
        '''
        best_idx = None
        best_effect = None
        num_nonnegative_unvisited = 0

        grad = self.calc_grad().flatten()
        for v in range(self.n**2):
            if self.is_flip_visited(v):
                if self.verbose:
                    print "v = %d, VISITED" %(v)
                continue
            effect = self.flip_gradient(v, grad)
            if not np.isinf(effect) and effect > -self.zthresh:
                num_nonnegative_unvisited += 1
            if not np.isinf(effect) \
                    and (best_effect is None or effect > best_effect):
                best_idx = v
                best_effect = effect

            if self.verbose:
                print "v = %d, effect = %.4f" %(v, effect)
        #end for
        return best_idx, best_effect, num_nonnegative_unvisited
    # end def}}}
    def eval_vertex(self): # {{{
        prod = self.calc_prod()
        
        for constr_set, constraints in enumerate(self.extra_constr):
            if len(constraints) == 0:
                continue
            # extra_constr is an array of tuples (yindex, coeff).
            # We need to have sum(prod[x, y] * coeff) == required_total, where
            # the row x is the constr_set ranging (0, n) and the column is given
            # by this particular constraint (along with coeff for that # column).
            if self.verbose:
                print "constr_set = %d, constraints = %s" %(constr_set, constraints)

            # Each constr is set of values of the form: (uyvar, prodmul, coeff)
            for constr_num, constr in enumerate(constraints):
                # Note: even numbered slackvar = 1 - prod
                #       odd numbered slackvar = 1 + prod
                # overall value of slackvar is 1+uy or 1-uy, and tup[1] toggles.
                # then multiplty slackvar * coeff, which is tup[2]
                total = reduce(lambda acc, tup:
                        acc + (1 + prod[constr_set, tup[0]] * tup[1]) * tup[2],
                        constr,
                        0)
                if abs(total - 2.0) > 1e-9:
                    print "check fail: constr_set = %d, constr_num = %d: %.6f" %(
                            constr_set, constr_num, total)
                    return False
                # end if
            # end for
        # end for

        prod_full = self.calc_prod_full()
        if self.verbose:
            print "eval_vertex constraint check, prod = \n%s" %(
                    np.round(prod_full))

        feasible = np.max(np.abs(prod_full)) < 1 + self.zthresh
        return feasible
    # end def}}}
    def flip_gradient(self, v, grad, debug = False): # {{{
        base = v / self.n
        total_effect = 0.0
        for u in range(self.n * base, self.n * (base + 1)):
            # self.urows[u][0] contains which +base+ row has this U-var.  Here
            # we need the coefficient sign, not the X-variable value sign.
            row = self.urows[u][0]
            sign = self.rows[row, 2*u]

            # Get coefficient of the slackvar that will go 0 -> 2.
            coeff = self.rows[row, self.vmap[v]]

            # Calculate the effect of moving this slackvar on the U-value.  Then
            # multiply this delta by the gradient.
            delta_u = -2 * coeff * sign
            obj_effect = delta_u * grad[u]
            total_effect += obj_effect
            if debug:
                print "u = %d, coeff = %f, sign = %d, delta = %f, grad = %f (%f)" %(
                    u, coeff, sign, delta, grad[u], obj_effect)
        return total_effect
    #end def # }}}
    def flip(self, v): # {{{
        '''
        Flip the variable U_v between {-1, +1}.  Since the variable U_v is
        represented as the difference X_{2v} - X_{2v+1}, this amounts to
        changing which of these variables is basic, so the variable that
        currently has value 0 will be the entering variable, and the one with
        value 2 will be the leaving variable.
        We are flipping the value of UY at flat index v, where flat indices are
        in row-major order.
        '''
        # The self.vmap holds the index v such that x_v is the currently 0 slack
        # variable that needs to be flipped to 2 in order to make the pivot.
        old_zero = self.vmap[v]
        new_zero = old_zero ^ 0x1
        self.vmap[v] = new_zero
        self.X[old_zero] = 2.0
        self.X[new_zero] = 0.0

        # Now in order to exchange, must pivot each of the (n) U values?
        base = v / self.n
        for i in range(self.n * base, self.n * (base + 1)):
            # self.urows holds which row contains U_i as a basic var.
            maniprow, xplus, xminus = self.urows[i]
            if self.verbose:
                print "i = %d, maniprow = %d" %(i, maniprow)

            # Each maniprow will have the old_zero variable: get coefficient.
            old_zero_coeff = self.rows[maniprow, old_zero]
            #print "%d: old_zero_coeff = %f" %(maniprow, old_zero_coeff)

            # Execute Pivot.
            self.rows[maniprow, -1] -= old_zero_coeff * 2.0
            self.rows[maniprow, old_zero] = 0.0
            self.rows[maniprow, new_zero] = -old_zero_coeff

            if self.rows[maniprow, -1] >= 0.0:
                self.X[xplus] = self.rows[maniprow, -1]
                self.X[xminus] = 0.0
            else:
                self.X[xplus] = 0.0
                self.X[xminus] = -self.rows[maniprow, -1]
        # end for
    # end def}}}
    def is_trapped(self): # {{{
        '''
        Return True if currently trapped, defined as being at a vertex such that
        every neighbor is: 1) invalid, i.e. ||UY|| > 1; or 2) gradient strictly
        negative.
        Also must be the first visited vertex: i,e. result of initial BFS finder.
        '''
        # Take a snapshot to use.
        self.snapshot()
        jump_exists = False
        grad = self.calc_grad().flatten()
        best = None
        worst = None

        for v in range(len(grad)):
            effect = self.flip_gradient(v, grad)
            if best is None or effect > best: best = effect
            if worst is None or effect < worst: worst = effect

            if effect < -self.zthresh:
                continue
            else:
                self.flip(v)
                if self.eval_vertex():
                    jump_exists = True
                self.restore()
                if jump_exists:
                    break
            # end if
        # end for

        self.restore(pop = True)
        return (not jump_exists
                and (
                        (self.n == 5 
                        #and np.abs(worst + 1.0) < 1e-3 # XXX: correct?
                        and len(self.visited) <= 25
                        )
                     or (self.n == 6
                        #and np.abs(worst + 1.0) < 1e-3
                        and len(self.visited) <= 228
                        )
                    )
                )
    # end def}}}

    def calc_obj(self, setU = True): # {{{
        if setU:
            self.__setU__()

        try:
            return np.log(abs(np.linalg.det(self.U)))
        except np.linalg.LinAlgError:
            return None
    # end def # }}}
    def calc_grad(self, setU = True): # {{{
        if setU:
            self.__setU__()

        try:
            return np.linalg.inv(self.U).transpose()
        except np.linalg.LinAlgError:
            return None
    # end def}}}
    def calc_prod(self, setU = True): # {{{
        if setU:
            self.__setU__()
        return self.U.dot(self.Y)
    # end def}}}
    def calc_prod_full(self, setU = True): # {{{
        if setU:
            self.__setU__()
        return self.U.dot(self.Yfull)
    # end def}}}
    def __setU__(self): # {{{
        '''
        Reconstruct U from first 2 * self.n**2 entries of X.
        '''
        for i in range(self.n):
            for j in range(self.n):
                xplus = self.X[2 * (self.n * i + j)]
                xminus = self.X[2 * (self.n * i + j) + 1]
                self.U[i, j] = xplus - xminus
            # end
        # end
    # end def # }}}
    def calc_v_det(self): # {{{
        v_mtx = np.matrix([2 * (v & 0x1) - 1 for v in self.vmap]) \
            .reshape(self.n, self.n)

        v_det = np.round(np.linalg.det(v_mtx))
        print "v_mtx = \n%s\ndet = %d" %(v_mtx, v_det)
        return v_det
    # end def}}}
 
    def is_done(self): # {{{
        if self.reduced:
            prod_full = self.calc_prod_full()
            full_bfs = np.max(np.abs(np.abs(prod_full) - 1)) < self.zthresh
            return full_bfs
        # end if

        return self.done_funcs[self.n]()
    # end def}}}
    def set_done_funcs(self): # {{{
        self.done_funcs = [
            None, 
            None,
            self.is_done_2,
            self.is_done_3,
            self.is_done_4,
            self.is_done_5,
            self.is_done_6,
            self.is_done_7,
            self.is_done_8,
        ]
    # end def}}}
    def is_done_2(self): # {{{
        return np.abs(np.linalg.det(self.U)) > self.zthresh
    # end def}}}
    def is_done_3(self): # {{{
        return np.abs(np.linalg.det(self.U)) > self.zthresh
    # end def}}}
    def is_done_4(self): # {{{
        grad = self.calc_grad().flatten()
        highest = None
        highest_abs = None
        for ef in [self.flip_gradient(v, grad) for v in range(self.n**2)]:
            if highest is None or ef > highest:
                highest = ef
            if highest_abs is None or abs(ef) > highest_abs:
                highest_abs = abs(ef)
        # end for
        if self.verbose:
            print "highest = %f, highest_abs = %f" %(highest, highest_abs)

        if highest < -0.50 + self.zthresh:
            return True

        return False
    # end def}}}
    def is_done_5(self): # {{{
        grad = self.calc_grad().flatten()

        highest = None
        highest_abs = None
        for ef in [self.flip_gradient(v, grad) for v in range(self.n**2)]:
            if highest is None or ef > highest:
                highest = ef
            if highest_abs is None or abs(ef) > highest_abs:
                highest_abs = abs(ef)
        # end for
        if self.verbose:
            print "highest = %f, highest_abs = %f" %(highest, highest_abs)

        if highest < -self.zthresh:
            return True

        return False
    # end def}}}
    def is_done_6(self): # {{{
        # TODO: Need to account for local optima.
        grad = self.calc_grad().flatten()

        highest = None
        highest_abs = None
        for v, ef in enumerate([self.flip_gradient(v, grad) \
                for v in range(self.n**2)]):
            if self.verbose:
                print "v = %d, effect = %.4f" %(v, ef)
            if highest is None or ef > highest:
                highest = ef
            if highest_abs is None or abs(ef) > highest_abs:
                highest_abs = abs(ef)
        # end for
        if self.verbose:
            print "highest = %f, highest_abs = %f" %(highest, highest_abs)

        #if highest < -self.zthresh:
        #    if self.n == self.k:
        #        UY = self.calc_prod_full()
        #        print "det = %.3f" %(np.linalg.det(UY))
        #    return True
        if np.abs(highest + 0.2) < 1e-2:
            return True

        return False
    # end def}}}
    def is_done_7(self): # {{{
        # TODO: Need to account for local optima.
        grad = self.calc_grad().flatten()

        highest = None
        highest_abs = None
        for ef in [self.flip_gradient(v, grad) for v in range(self.n**2)]:
            if highest is None or ef > highest:
                highest = ef
            if highest_abs is None or abs(ef) > highest_abs:
                highest_abs = abs(ef)
        # end for
        if self.verbose:
            print "highest = %f, highest_abs = %f" %(highest, highest_abs)

        if highest < -self.zthresh:
            return True

        return False
    # end def}}}
    def is_done_8(self): # {{{
        # TODO: Need to account for local optima.
        grad = self.calc_grad().flatten()

        highest = None
        highest_abs = None
        for ef in [self.flip_gradient(v, grad) for v in range(self.n**2)]:
            if highest is None or ef > highest:
                highest = ef
            if highest_abs is None or abs(ef) > highest_abs:
                highest_abs = abs(ef)
        # end for
        if self.verbose:
            print "highest = %f, highest_abs = %f" %(highest, highest_abs)

        if highest < -self.zthresh:
            return True

        return False
    # end def}}}

    def is_certain_max_5(self): # {{{
        if self.n != 5:
            return False

        grad = self.calc_grad().flatten()
        highest = None
        lowest = None
        for ef in [self.flip_gradient(v, grad) for v in range(self.n**2)]:
            if highest is None or ef > highest:
                highest = ef
            if lowest is None or ef < lowest:
                lowest = ef
        # end for
        
        if np.abs(highest + 1.0 / 3.0) <= 1e-6 \
                and np.abs(lowest + 2.0 / 3.0) <= 1e-6:
            return True

        return False
    # end def}}}
# end class FlexTab}}}
# end Tableau classes}}}

if __name__ == '__main__': #{{{
    #tester_clean()
    #tester_badcols()
    #many_func(n8, reps = 100, stop_on_error = False, verbose = False)
    #bad_counter()
    #n5_baseplus()
    #n5_specific(reps_per_mtx = 10)
    #n5_gen_trap()
    #n5_retry_trap()
    #trap_experiment()
    second_trap_experiment()
#end main }}}
