import numpy as np
import itertools
import math

class FeasibleRegion:
    '''
    Contains the raw constraints and an orthonormal basis for the constraints.
    Inserting a row automatically performs the fast update
    Removing a row updates from scratch

    self.b holds raw rows of Y
    self.p holds orthonomoralized rows of Y
    self.col_map holds mapping of columns of UY <-> rows of b
    '''
    def __init__(self, Y, tol=1e-9):
        self.Y = np.matrix(np.copy(Y))
        (self.n, self.k) = Y.shape
        self.b = [np.matrix(np.zeros( (0,self.n) )) for i in range(self.n)]
        self.p = [np.matrix(np.zeros( (0,self.n) )) for i in range(self.n)]
        self.col_map = [[] for i in range(self.n)]
        self.tol = tol

    def __repr__(self):
        ret = 'Y = %s\n' % self.Y
        ret += 'b = \n'
        for i in range(self.n):
            ret += '%s\n' % self.b[i]

        ret += 'p = \n'
        for i in range(self.n):
            ret += '%s\n' % self.p[i]

        ret += 'col_map = \n'
        for i in range(self.n):
            ret += '%s\n' % self.col_map[i]
        
        return ret

    def insert_mtx( self, update ):
        (nn,kk) = update.shape
        for i in range(nn):
            for j in range(kk):
                if update[i,j] == True:
                    self.insert( i, j )

    def remove_mtx( self, update ):
        (nn,kk) = update.shape
        for i in range(nn):
            for j in range(kk):
                if update[i,j] == True:
                    self.remove( i, j, False )

        self.__recompute_p()

    def insert( self, row, column ):
        '''
        Call when a constraint becomes active.  If UY_{i,j} becomes \pm 1
        then call this: insert( i, j )

        This grabs the appropriate column of Y and sticks it in the appropriate
        matrix B.  Then P is updated according the our optimized gram-schmidt
        process.
        '''
        #If this is true the entry is already in the data structure
        #Could consider raising an exception
        if column in self.col_map[row]:
            return

        new_row = np.matrix(np.copy(self.Y[:,column].T))
        self.b[row] = np.concatenate( (self.b[row], new_row) )
        self.p[row] = np.concatenate( (self.p[row], new_row) )
        self.col_map[row].append( column )
        self.__reorthonormalize_p(row)

    def remove( self, row, column, update_p = True ):
        '''
        Opposite of above function - the appropriate entry is removed
        If update_p is true, then p is recomputed from scratch.  Expensive, but
        we typically remove an arbitrary constraint so it's hard to have a more
        general optimal way to do this.  

        If update_p is false, p is not updated, this is useful if you have to
        remove several entries at once
        '''
        col_idx = self.col_map[row].index(column) # will raise ValueError if bad
        self.b[row] = np.delete( self.b[row], col_idx, 0 )
        self.col_map[row].remove(column)

        if update_p:
            self.__recompute_p( row )

    def reject_mtx( self, V, t1=1e-12 ):
        res = np.matrix(np.zeros((0,self.n)))
        for i in range(self.n):
            v = self.reject_vec( V[i,:].T, i )
            norm = np.linalg.norm(v)

            if norm < t1:
                #If norm is below t1, set to zero
                v = np.matrix(np.zeros( (1, self.n) ) )

            res = np.concatenate( (res, v) ) 
            
        return res

    def reject_vec( self, v, row ):
        for i in range(self.p[row].shape[0]):
            p = self.p[row][i,:]
            try:
                s = (p * v)/(p * p.T)
            except IndexError:
                print "i = %d, row = %d" % (i, row)
                print self
                raise IndexError

            s = s * p
            v = v - s.T

        return v.T

    def __reorthonormalize_p( self, row ):
        '''
        Assumes that only one row of p has been added! otherwise call 
        __recompute_p to do full orthonormalization
        '''
        pp = self.p[row]
        (l, _) = pp.shape

        if l == 1:
            self.p[row] = pp / np.linalg.norm(pp)
            return

        new_row = pp[-1,:]

        for i in range(l-1):
            #Perform vector rejection.
            u = pp[i,:]
            v = new_row / np.linalg.norm(new_row)
            vv = v - ((u*v.T)/(u*u.T)) * u
            norm = np.linalg.norm(vv)
            if norm < self.tol:
                #Constraint is redundant so delete it
                self.p[row] = np.delete( self.p[row], l-1,0 )
                return
            else:
                #Normalize and add
                vv /= norm
                new_row = vv

        self.p[row][-1,:] = new_row


    def __recompute_p(self, row = -1):
        '''
        If row is -1 then recompute all blocks of p.  Otherwise just recompute
        one block of p from the corresponding block of b
        '''
        if row == -1:
            for i in range(self.n):
                self.__recompute_p( i )

        pp = np.matrix(np.copy(self.b[row]))

        A = pp * pp.T
        (nn,_) = A.shape

        #This means the corresponding entry of b is empty
        if nn == 0:
            self.p[row] = np.matrix(np.zeros( (0,self.n) ))
            return

        pp[0,:] /= np.linalg.norm(pp[0,:])

        bad = []
        for i in range(nn):
            for j in range(nn):
                if j <= i:
                    continue
                if abs(A[i,j]) > self.tol:
                    if not ( i in bad ):
                        bad.append(i)
                    if not ( j in bad ):
                        bad.append(j)

        #print "Non-orthogonal vectors: %s" % bad

        zero = []
        for (i,j) in itertools.combinations(bad,2):
            if (i in zero) or (j in zero):
                continue
            #Perform vector rejection
            u = pp[i,:]
            v = pp[j,:]
            vv = v - ((u*v.T)/(u*u.T))*u
            norm = np.linalg.norm(vv)
            if norm < self.tol:
                if not (j in zero):
                    zero.append(j)
            else:
                pp[j,:] = vv / norm

        zero.sort()
        #print "Redundant contraints: %s" % zero
        pp = np.delete(pp,zero,0)

        self.p[row] = pp
