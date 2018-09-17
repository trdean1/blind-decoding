// includes {@
extern crate nalgebra as na;

use std::ops::{Index, IndexMut};
use std::fmt;
// end @}

//{@
// Matrix structures as a set of blocks along its diagonal, where each block is n x k, so that
// entire matrix is qn x qk for some q \in Z.  Each block is assumed to be dense.  Each
// off-diagonal entry is automatically assumed to be 0.
//@}
#[derive(Clone)]
pub struct DiagBlockMatrix { //{@
    pub block_nrows: usize,
    pub block_ncols: usize,
    pub nrows: usize,
    pub ncols: usize,
    pub blocks: Vec<na::DMatrix<f64>>, /*Vec<RowMatrix>,*/
    pub inv_block_nrows: f64,
    pub inv_block_ncols: f64,
} //@}
impl DiagBlockMatrix { //{@
    pub fn new(block_nrows: usize, block_ncols: usize, num_blocks: usize) -> DiagBlockMatrix { //{@
        DiagBlockMatrix {
            block_nrows,
            block_ncols,
            nrows: block_nrows * num_blocks,
            ncols: block_ncols * num_blocks,
            blocks: vec![na::DMatrix::<f64>::zeros(block_nrows, block_ncols); num_blocks],
            //blocks: vec![RowMatrix::zeros(block_nrows, block_ncols); num_blocks],
            inv_block_nrows: 1.0 / block_nrows as f64,
            inv_block_ncols: 1.0 / block_ncols as f64,
        }
    } //@}

    fn xlat<'a>(&'a self, mut i: usize, mut j: usize) -> (usize, usize, usize, usize) { //{@
        let mut i_block = 0;
        while i >= self.block_nrows {
            i_block += 1;
            i -= self.block_nrows;
        }
        let mut j_block = 0;
        while j >= self.block_ncols {
            j_block += 1;
            j -= self.block_ncols;
        }
        (i_block, j_block, i, j)
    } //@}
    pub fn add_row_multiple(&mut self, dst: usize, src: usize, mult: f64) { //{@
        /*
        let src_block = src / self.block_nrows;
        let dst_block = dst / self.block_nrows;
        assert!(src_block == dst_block && src_block < self.blocks.len(),
                "DiagBlockMatrix: add_row_multiple bad index");

        let block = &mut self.blocks[src_block];
        let src_row_idx = src % self.block_nrows;
        let dst_row_idx = dst % self.block_nrows;
        */

        let mut src_block = 0;
        let mut src_row_idx = src;
        while src_row_idx >= self.block_nrows {
            src_block += 1;
            src_row_idx -= self.block_nrows;
        }
        let mut dst_row_idx = dst;
        while dst_row_idx >= self.block_nrows {
            dst_row_idx -= self.block_nrows;
        }
        let block = &mut self.blocks[src_block];

        for j in 0 .. self.block_ncols {
            block[(dst_row_idx,j)] += mult * block[(src_row_idx,j)];
        }

    } //@}
    pub fn div_row_float(&mut self, row: usize, divisor: f64) { //{@
        assert!(row < self.nrows, "DiagBlockMatrix: row index bad");
        if divisor == 1.0 {
            return;
        }
        let mut block_idx = 0;
        let mut row_within_block = row;
        while row_within_block >= self.block_nrows {
            block_idx += 1;
            row_within_block -= self.block_nrows;
        }
        /*
        let block_idx = row / self.block_nrows;
        let row_within_block = row % self.block_nrows;
        */

        //let mut raw_row = self.blocks[block_idx].row_mut(row_within_block);
        //raw_row /= divisor;
        for j in 0 .. self.block_ncols {
            self.blocks[block_idx][(row_within_block,j)] /= divisor;
        }
        /*
        let raw_row = &mut self.blocks[block_idx][row_within_block];
        for j in 0 .. self.block_ncols {
            raw_row[j] /= divisor;
        }
        */
    } //@}
    pub fn to_dmatrix(&self) -> na::DMatrix<f64> { //{@
        let mut mtx = na::DMatrix::<f64>::zeros(self.nrows, self.ncols);
        for bnum in 0 .. self.blocks.len() {
            let base_row = bnum * self.block_nrows;
            let base_col = bnum * self.block_ncols;
            for i in base_row .. base_row + self.block_nrows {
                for j in base_col .. base_col + self.block_ncols {
                    mtx[(i, j)] = self[(i, j)];
                }
            }
        }
        mtx
    } //@}
    pub fn index_block<'a>(&'a self, n: usize, i: usize, j: usize) -> &'a f64 { //{@
        &self.blocks[n][(i, j)] 
    } //@}

    pub fn index_block_mut<'a>(&'a mut self, n: usize, i: usize, j: usize) 
        -> &'a mut f64 { //{@
        &mut self.blocks[n][(i, j)] 
    } //@}
} //@}
const ZERO_F64: f64 = 0.0;
impl Index<(usize, usize)> for DiagBlockMatrix { //{@
    type Output = f64;
    fn index<'a>(&'a self, ij: (usize, usize)) -> &'a f64 {
        /*
        match self.xlat(ij.0, ij.1) {
            Ok(f) => match f {
                Some(f) => f,
                //None => &0.0,
                None => &ZERO_F64,
            },
            Err(e) => panic!(e),
        }
        */
        let (i_block, j_block, i, j) = self.xlat(ij.0, ij.1);

        if i_block != j_block {
            &ZERO_F64
        } else {
            &self.blocks[i_block][(i, j)]
        }
    }
} //@}
impl IndexMut<(usize, usize)> for DiagBlockMatrix { //{@
    fn index_mut<'a>(&'a mut self, ij: (usize, usize)) -> &'a mut f64 {
        /*
        match self.xlat_mut(ij.0, ij.1) {
            Ok(f) => match f {
                Some(mut f) => f,
                None => panic!("DiagBlockMatrix: Attempt to set off-diag elt"),
            },
            Err(e) => panic!(e),
        }
        */
        let (i_block, j_block, i, j) = self.xlat(ij.0, ij.1);

        if i_block != j_block {
            panic!("DiagBlockMatrix: index mut off diagonal");
        } else {
            &mut self.blocks[i_block][(i, j)]
        }
    }
} //@}
impl fmt::Display for DiagBlockMatrix { //{@
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mtx = self.to_dmatrix();
        write!(f, "{:.3}", mtx)
    }
} //@}

#[derive(Clone)]
pub struct SparseTMatrix { //{@
    pub n: usize,
    pub k: usize,
    nrows: usize,
    ncols: usize,
    block: DiagBlockMatrix,
    //sparse: SparseVecMatrix,
    sparse: DiagBlockMatrix,
    rightcol: Vec<f64>,
} //@}
impl SparseTMatrix { //{@
    pub fn new(n: usize, k: usize) -> SparseTMatrix { //{@
        let nrows = 2 * n * k;
        let ncols = nrows + 2 * n * n + 1;
        let block = DiagBlockMatrix::new(2 * k, 2 * n, n);
        //let sparse = SparseVecMatrix::new(nrows, nrows);
        let sparse = DiagBlockMatrix::new(2 * k, 2 * k, n);
        let rightcol = vec![1.0; nrows];
        SparseTMatrix { n, k, nrows, ncols, block, sparse, rightcol }
    } //@}
    pub fn nrows(&self) -> usize { //{@
        self.nrows
    } //@}
    pub fn ncols(&self) -> usize { //{@
        self.ncols
    } //@}
    pub fn add_row_multiple(&mut self, dst: usize, src: usize, mult: f64) { //{@
        assert!(dst < self.nrows && src < self.nrows, "SparseTMatrix: bad row index");
        self.block.add_row_multiple(dst, src, mult);
        self.sparse.add_row_multiple(dst, src, mult);
        if mult != 1.0 {
            self.rightcol[dst] += self.rightcol[src] * mult;
        } else {
            self.rightcol[dst] += self.rightcol[src];
        }
    } //@}
    pub fn div_row_float(&mut self, row: usize, divisor: f64) { //{@
        assert!(row < self.nrows, "SparseTMatrix: bad row index");
        self.block.div_row_float(row, divisor);
        self.sparse.div_row_float(row, divisor);
        if divisor != 1.0 {
            self.rightcol[row] /= divisor;
        }
    } //@}
    // TODO: Some sort of fmt_constraints that returns a String
    pub fn index_block<'a>(&'a self, n: usize, i: usize, j: usize) -> &'a f64 { //{@
        self.block.index_block(n, i, j)
    } //@}
    pub fn index_sparse<'a>(&'a self, n: usize, i: usize, j: usize) -> &'a f64 { //{@
        self.sparse.index_block(n, i, j)
    } //@}

    pub fn index_block_mut<'a>(&'a mut self, n: usize, i: usize, j: usize) 
        -> &'a mut f64 { //{@
        self.block.index_block_mut(n, i, j)
    } //@}
    pub fn index_sparse_mut<'a>(&'a mut self, n: usize, i: usize, j: usize) 
        -> &'a mut f64 { //{@
        self.sparse.index_block_mut(n, i, j)
    } //@}

    pub fn index_rhs<'a>(&'a self, n: usize, row: usize) -> &'a f64 {
        &self.rightcol[2*n*self.k + row]
    }

    pub fn index_rhs_mut<'a>(&'a mut self, n: usize, row: usize) -> &'a mut f64 {
        &mut self.rightcol[2*n*self.k + row]
    }
    
    pub fn to_dmatrix(&self) -> na::DMatrix<f64> { //{@
        // create zeros matrix
        let mut mtx = na::DMatrix::<f64>::zeros(self.nrows, self.ncols);

        // copy in block, sparse, rightcol
        for j in 0 .. self.block.ncols {
            for i in 0 .. self.nrows {
                mtx[(i, j)] = self.block[(i, j)];
            }
        }
        for j in 0 .. self.sparse.ncols {
            for i in 0 .. self.nrows {
                mtx[(i, j + self.block.ncols)] = self.sparse[(i, j)];
            }
        }
        for i in 0 .. self.nrows {
            mtx[(i, self.ncols - 1)] = self.rightcol[i];
        }
        mtx
    } //@}
    // TODO: Maybe some means of initializing from a DMatrix?  Or anything else that makes sense
    // for the larger program?
} //@}
impl fmt::Display for SparseTMatrix { //{@
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mtx = self.to_dmatrix();
        write!(f, "{:.3}", mtx)
    }
} //@}
impl Index<(usize, usize)> for SparseTMatrix { //{@
    type Output = f64;
    fn index<'a>(&'a self, ij: (usize, usize)) -> &'a f64 {
        //assert!(ij.0 < self.nrows && ij.1 < self.ncols, "SparseTMatrix index out of range");
        if ij.1 < self.block.ncols {
            &self.block[ij]
        } else if ij.1 < self.ncols - 1 {
            &self.sparse[(ij.0, ij.1 - self.block.ncols)]
        } else if ij.1 == self.ncols - 1 {
            &self.rightcol[ij.0]
        } else {
            panic!("SparseTMatrix index out of range");
        }
    }
} //@}
impl IndexMut<(usize, usize)> for SparseTMatrix { //{@
    fn index_mut<'a>(&'a mut self, ij: (usize, usize)) -> &'a mut f64 {
        //assert!(ij.0 < self.nrows && ij.1 < self.ncols, "SparseTMatrix index out of range");
        if ij.1 < self.block.ncols {
            &mut self.block[ij]
        } else if ij.1 < self.ncols - 1 {
            &mut self.sparse[(ij.0, ij.1 - self.block.ncols)]
        } else if ij.1 == self.ncols - 1 {
            &mut self.rightcol[ij.0]
        } else {
            panic!("SparseTMatrix index out of range");
        }
    }
} //@}

#[cfg(test)]
mod tests { //{@
    use super::DiagBlockMatrix;
    #[test]
    fn diag_block_test() {
        let mut dbmtx = DiagBlockMatrix::new(2, 2, 4);
        dbmtx[(0, 0)] = 1.0;
        dbmtx[(1, 0)] = 3.0;
        dbmtx[(0, 1)] = 2.0;
        println!("{}", dbmtx);
    }
} //@}
