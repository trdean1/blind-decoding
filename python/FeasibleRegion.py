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
        self.Y = Y
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

    def insert_by_mtx( self, update ):
        for i in range(self.n):
            for j in range(self.k):
                if update[i,j] == True:
                    self.insert( i, j )

    def remove_by_mtx( self, update ):
        for i in range(self.n):
            for j in range(self.k):
                if update[i,j] == True:
                    self.remove( i, j, False )

        self.__recompute_p()

    def insert( self, row, column ):
        if column in self.col_map[row]:
            return

        new_row = self.Y[:,column].T
        self.b[row] = np.concatenate( (self.b[row], new_row) )
        self.p[row] = np.concatenate( (self.p[row], new_row) )
        self.col_map[row].append( column )
        self.__reorthonormalize_p(row)

    def remove( self, row, column, update_p = True ):
        col_idx = self.col_map[row].index(column) # will raise ValueError if bad
        self.b[row] = np.delete( self.b[row], col_idx, 0 )
        self.col_map[row].remove(column)

        if update_p:
            self.__recompute_p( row )

    def get_p_block( self, row ):
        return self.p[row]

    def reject_mtx( self, V ):
        for i in range(self.n):
            V[i,:] = reject_vec( V[i,:], i )

        return V

    def reject_vec( self, v, row ):
        for i in range(len(self.col_map[row])):
            s =  (self.p[row][i,:] * v) / ( self.p[row][i,:] * self.p[row][i,:].T)
            s = s*p[row][i,:]
            v = v - s.T

        return v

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
            #Perform vector rejection.  u is already normalized
            u = pp[i,:]
            v = new_row
            vv = v - (u*v.T) * u
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

        pp = self.p[row]
        A = pp * pp.T

        #Find non-zero entires in A, should be easier using a sparse
        #implementation
        bad = []
        for i in range(self.n):
            for j in range(self.n):
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
            u = pp[i,:]
            v = pp[j,:]
            vv = v - ((u*v.T)/(u*u.T))*u
            if np.linalg.norm(vv) < tol:
                if not (j in zero):
                    zero.append(j)
            else:
                pp[j,:] = vv

        zero.sort()
        #print "Redundant contraints: %s" % zero
        pp = np.delete(pp,zero,0)

        self.p[row] = pp
