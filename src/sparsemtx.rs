// includes {@
extern crate nalgebra as na;

use std::collections::{HashMap};
use std::ops::{Index, IndexMut};
use std::fmt;
use std::cmp::Ordering;
// end @}

#[derive(Clone)]
pub struct RowMatrix { //{@
    nrows: usize,
    ncols: usize,
    elts: Vec<Vec<f64>>,
} //@}
impl RowMatrix { //{@
    fn zeros(nrows: usize, ncols: usize) -> RowMatrix {
        RowMatrix {
            nrows,
            ncols,
            elts: vec![ vec![0.0; ncols]; nrows],
        }
    }
} //@}
impl Index<usize> for RowMatrix { //{@
    type Output = Vec<f64>;
    fn index<'a>(&'a self, i: usize) -> &'a Vec<f64> {
        &self.elts[i]
    }
} //@}
impl IndexMut<usize> for RowMatrix { //{@
    fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut Vec<f64> {
        &mut self.elts[i]
    }
} //@}
impl Index<(usize, usize)> for RowMatrix { //{@
    type Output = f64;
    fn index<'a>(&'a self, ij: (usize, usize)) -> &'a f64 {
        &self.elts[ij.0][ij.1]
    }
} //@}
impl IndexMut<(usize, usize)> for RowMatrix { //{@
    fn index_mut<'a>(&'a mut self, ij: (usize, usize)) -> &'a mut f64 {
        &mut self.elts[ij.0][ij.1]
    }
} //@}
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
    fn xlat_calc(&self, mut i: usize, mut j: usize) //{@
            -> Result<Option<(usize, usize, usize)>, String>
    {
        if i >= self.nrows || j >= self.ncols {
            return Err(format!("DiagBlockMatrix: i = {}, nrows = {}, j = {}, ncols = {}",
                    i, self.nrows, j, self.ncols));
        }
        // determine which block the i and j indices correspond to
        /*
        /*
        let i_block = i / self.block_nrows;
        let j_block = j / self.block_ncols;
        */
        let i_block = (i as f64 * self.inv_block_nrows) as usize;
        let j_block = (j as f64 * self.inv_block_ncols) as usize;

        // if i and j don't correspond to same block, then this entry is off-diagonal
        if i_block != j_block {
            return Ok(None); 
        }
        // return the entry from within the dense block
        let i_idx = i % self.block_nrows;
        let j_idx = j % self.block_ncols;
        Ok(Some((i_block, i_idx, j_idx)))
        */
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

        if i_block != j_block {
            Ok(None)
        } else {
            Ok(Some((i_block, i, j)))
        }
    } //@}
    fn xlat<'a>(&'a self, i: usize, j: usize) -> Result<Option<&'a f64>, String> { //{@
        match self.xlat_calc(i, j)? {
            Some((b_idx, i_idx, j_idx)) => Ok(Some(&self.blocks[b_idx][(i_idx, j_idx)])),
            None => Ok(None),
        }
    } //@}
    fn xlat_mut<'a>(&'a mut self, i: usize, j: usize) -> Result<Option<&'a mut f64>, String> { //{@
        match self.xlat_calc(i, j)? {
            Some((b_idx, i_idx, j_idx)) => Ok(Some(&mut self.blocks[b_idx][(i_idx, j_idx)])),
            None => Ok(None),
        }
    } //@}
    fn xlat_2<'a>(&'a self, mut i: usize, mut j: usize) -> (usize, usize, usize, usize) { //{@
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
        let mut dst_block = 0;
        let mut dst_row_idx = dst;
        while dst_row_idx >= self.block_nrows {
            dst_block += 1;
            dst_row_idx -= self.block_nrows;
        }
        let block = &mut self.blocks[src_block];

        //let addend = block.row(src_row_idx) * mult;
        //let mut tgt = block.row_mut(dst_row_idx);
        //tgt += addend;
        for j in 0 .. self.block_ncols {
            block[(dst_row_idx,j)] += mult * block[(src_row_idx,j)];
        }

        /*
        for j in 0 .. self.block_ncols {
            if mult != 1.0 {
                block[(dst_row_idx, j)] += block[(src_row_idx, j)] * mult;
            } else {
                block[(dst_row_idx, j)] += block[(src_row_idx, j)];
            }
        }
        */
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
        let (i_block, j_block, i, j) = self.xlat_2(ij.0, ij.1);

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
        let (i_block, j_block, i, j) = self.xlat_2(ij.0, ij.1);

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
pub struct SparseVecMatrix { //{@
    pub nrows: usize,
    pub ncols: usize,
    pub entries: Vec<Vec<(usize, f64)>>,
} //@}
impl SparseVecMatrix { //{@
    pub fn new(nrows: usize, ncols: usize) -> SparseVecMatrix { //{@
        SparseVecMatrix {
            nrows,
            ncols,
            entries: vec![vec![]; nrows],
        }
    } //@}
    fn get_unchecked<'a>(&'a self, i: usize, j: usize) -> &'a f64 { //{@
        match self.entries[i].binary_search_by(|&(elt_j, _)| elt_j.cmp(&j)) {
            Ok(phys_j) => &self.entries[i][phys_j].1,
            Err(_) => &0.0,
        }
    } //@}
    fn get_mut_unchecked<'a>(&'a mut self, i: usize, j: usize) -> &'a mut f64 { //{@
        match self.entries[i].binary_search_by(|&(elt_j, _)| elt_j.cmp(&j)) {
            Ok(phys_j) => &mut self.entries[i][phys_j].1,
            Err(phys_j) => {
                self.entries[i].insert(phys_j, (j, 0.0));
                &mut self.entries[i][phys_j].1
            },
        }
    } //@}
    pub fn add_row_multiple(&mut self, dst: usize, src: usize, mult: f64) { //{@
        assert!(dst < self.nrows && src < self.nrows,
                "SparseVecMatrix: add_row_multiple bad index");
        let mut new_row = vec![];
        let mut src_phys_j = 0;
        let mut dst_phys_j = 0;
        let non_ident_mult = mult != 1.0;
        while src_phys_j < self.entries[src].len() && dst_phys_j < self.entries[dst].len() {
            let &(src_virt_j, src_val) = &self.entries[src][src_phys_j];
            let &(dst_virt_j, dst_val) = &self.entries[dst][dst_phys_j];
            match src_virt_j.cmp(&dst_virt_j) {
                Ordering::Equal => {
                    let v = dst_val + if non_ident_mult { mult * src_val } else { src_val };
                    new_row.push((dst_virt_j, v));
                    src_phys_j += 1;
                    dst_phys_j += 1;
                },
                Ordering::Less => {
                    let v = if non_ident_mult { mult * src_val } else { src_val };
                    new_row.push((src_virt_j, v));
                    src_phys_j += 1;
                },
                Ordering::Greater => {
                    new_row.push((dst_virt_j, dst_val));
                    dst_phys_j += 1;
                },
            }
        }
        for &(virt_j, src_val) in self.entries[src][src_phys_j..].iter() {
            let v = if non_ident_mult { mult * src_val } else { src_val };
            new_row.push((virt_j, v));
        }
        new_row.extend_from_slice(&self.entries[dst][dst_phys_j..]);

        self.entries[dst] = new_row;
    } //@}
    pub fn div_row_float(&mut self, row: usize, divisor: f64) { //{@
        assert!(row < self.nrows, "DiagBlockMatrix: row index bad");
        for (_virt_j, val) in self.entries[row].iter_mut() {
            *val /= divisor;
        }
    } //@}
    pub fn to_dmatrix(&self) -> na::DMatrix<f64> { //{@
        let mut mtx = na::DMatrix::<f64>::zeros(self.nrows, self.ncols);
        for i in 0 .. self.nrows {
            for &(virt_j, val) in self.entries[i].iter() {
                mtx[(i, virt_j)] = val;
            }
        }
        mtx
    } //@}
} //@}
impl Index<(usize, usize)> for SparseVecMatrix { //{@
    type Output = f64;
    fn index<'a>(&'a self, ij: (usize, usize)) -> &'a f64 {
        assert!(ij.0 < self.nrows && ij.1 < self.ncols, "SparseVecMatrix index out of range");
        self.get_unchecked(ij.0, ij.1)
    }
} //@}
impl IndexMut<(usize, usize)> for SparseVecMatrix { //{@
    fn index_mut<'a>(&'a mut self, ij: (usize, usize)) -> &'a mut f64 {
        assert!(ij.0 < self.nrows && ij.1 < self.ncols, "SparseVecMatrix index out of range");
        self.get_mut_unchecked(ij.0, ij.1)
    }
} //@}
impl fmt::Display for SparseVecMatrix { //{@
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
    pub fn to_dmatrix(&self) -> na::DMatrix<f64> { //{@
        // create zeros matrix
        let mut mtx = na::DMatrix::<f64>::zeros(self.nrows, self.ncols);

        // get copies of block and spare matrices
        let block_full = self.block.to_dmatrix();
        let sparse_full = self.sparse.to_dmatrix();

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
