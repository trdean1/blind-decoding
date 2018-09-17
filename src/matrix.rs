extern crate test;

extern crate nalgebra as na;
extern crate rand;

use rand::distributions::{Normal, Distribution};
use rand::Rng;
use std;

use is_feasible;

#[allow(dead_code)]
//{@
///Generate a random Gaussian(0, 1) matrix of the given dimensions.
//@}
pub fn rand_matrix(nrows: usize, ncols: usize) -> na::DMatrix<f64> { //{@
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(nrows * ncols);
    let dist = Normal::new(0.0, 1.0);
    for _ in 0 .. (nrows * ncols) {
        data.push(dist.sample(&mut rng));
    }
    na::DMatrix::from_column_slice(nrows, ncols, &data)
} //@}

///Genreate a random Gaussian(0, 1) complex matrix that is n x n.
///represented as [re im; -im re]
#[allow(dead_code)]
pub fn rand_complex_matrix(n_tx: usize, m_rx: usize) -> na::DMatrix<f64> {
    let n_tx_b = n_tx / 2;
    let m_rx_b = m_rx / 2;
    let count_b = n_tx_b * m_rx_b;
    let count = n_tx * m_rx;

    let mut rng = rand::thread_rng();
    let mut real_data = Vec::with_capacity(count_b);
    let mut imag_data = Vec::with_capacity(count_b);
    let mut data = Vec::with_capacity(count);

    let dist = Normal::new(0.0, 1.0);
    for _ in 0 .. count_b {
        real_data.push(dist.sample(&mut rng));
        imag_data.push(dist.sample(&mut rng));
    }

    for i in 0 .. m_rx_b {
        data.extend( &real_data[n_tx_b * i .. n_tx_b * (i+1)] );
        data.extend( &imag_data[n_tx_b * i .. n_tx_b * (i+1)] );
    }

    imag_data.iter_mut().for_each(|x| *x *= -1.0);

    for i in 0 .. m_rx_b {
        data.extend( &imag_data[n_tx_b * i .. n_tx_b * (i+1)] );
        data.extend( &real_data[n_tx_b * i .. n_tx_b * (i+1)] );
    }

    na::DMatrix::from_column_slice(n_tx, m_rx, &data)
}

//{@
/// Generate a random orthogonal n x n matrix
//@}
fn random_orthogonal(n: usize) -> na::DMatrix<f64> { //{@
    na::QR::new(rand_matrix(n, n)).q()
} //@}

//{@
/// Find a random feasible point, which is an n x n U such that all entries of
/// UY are bounded by 1 in absolute value.
/// Input:   Y = n x k matrix of received sybest_state.mbols.
/// Output:  U = n x n feasible inverse of the channel gain matrix.
//@}
pub fn rand_init(y: &na::DMatrix<f64>) -> na::DMatrix<f64> { //{@
    let n = y.shape().0;
    let mut u = random_orthogonal(n);
    let mut scale = 1;
    while !is_feasible(&u, &y, None) {
        u = random_orthogonal(n);
        u /= scale as f64;
        scale <<= 1;
    }
    u
} //@}

//{@
/// Generate a random \pm 1 matrix of the given dimensions.
//@}
pub fn rand_pm1_matrix(nrows: usize, ncols: usize) -> na::DMatrix<f64> { //{@
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(nrows * ncols);
    for _ in 0 .. (nrows * ncols) {
        data.push(1.0 - (2 * rng.gen_range(0, 2)) as f64);
    }
    na::DMatrix::from_column_slice(nrows, ncols, &data)
} //@}

#[allow(dead_code)]
pub fn rand_unit(n: usize) -> na::DVector<f64> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n);
    let dist = Normal::new(0.0, 1.0);
    for _ in 0 .. (n) {
        data.push(dist.sample(&mut rng));
    }

    let mut v = na::DVector::from_column_slice(n, &data);
    let norm = v.norm();
    v /= norm;

    return v;
}

//{@
/// Randomly select one (n, k) value from the array of (n, k) pairs given in
/// +dims+.  Fill the necessary extra columns (if necessary) with random iid
/// \pm 1 entries.
//@}
pub fn get_matrix(dims: &[(usize, usize)]) -> na::DMatrix<f64> { //{@
    let xmats = vec![
        BaseMatrix { basemtx: na::DMatrix::from_row_slice(2, 2,
                &vec![  1.,  1.,
                        1., -1., 
                ]), extra_cols: 0,
        },
        BaseMatrix { basemtx: na::DMatrix::from_row_slice(3, 4,
                &vec![  1.,  1.,  1.,  1.,
                        1.,  1., -1., -1.,
                        1., -1.,  1., -1.,
                ]), extra_cols: 0,
        },
        BaseMatrix { basemtx: na::DMatrix::from_row_slice(4, 5,
                &vec![  1.,  1.,  1.,  1.,  1.,
                        1.,  1., -1., -1.,  1.,
                        1., -1.,  1., -1.,  1.,
                        1., -1., -1.,  1., -1.,
                ]), extra_cols: 0,
        },
        BaseMatrix { basemtx: na::DMatrix::from_row_slice(5, 6,
                &vec![  -1., -1., -1., -1., -1., -1.,
                        -1., -1., -1.,  1.,  1.,  1.,
                        -1., -1.,  1., -1.,  1.,  1.,
                        -1.,  1., -1., -1.,  1., -1.,
                        -1.,  1.,  1.,  1., -1., -1.,
                ]), extra_cols: 0,
        },
        BaseMatrix { basemtx: na::DMatrix::from_row_slice(6, 6,
                &vec![ -1.,  1.,  1.,  1.,  1.,  1.,
                        1., -1.,  1.,  1.,  1.,  1.,
                        1.,  1., -1.,  1.,  1.,  1.,
                       -1., -1., -1., -1.,  1.,  1.,
                       -1., -1., -1.,  1., -1.,  1.,
                       -1., -1., -1.,  1.,  1., -1.,
                ]), extra_cols: 0,
        },
        BaseMatrix { basemtx: na::DMatrix::from_row_slice(8, 8,
            &vec![  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                    1.,  1., -1., -1.,  1.,  1., -1., -1.,
                    1., -1., -1.,  1.,  1., -1., -1.,  1.,
                    1., -1.,  1., -1.,  1., -1.,  1., -1.,
                    1.,  1.,  1.,  1., -1., -1., -1., -1.,
                    1.,  1., -1., -1., -1., -1.,  1.,  1.,
                    1., -1., -1.,  1., -1.,  1.,  1., -1.,
                    1., -1.,  1., -1., -1.,  1., -1.,  1.,
            ]), extra_cols: 0,
        },
        BaseMatrix { basemtx: na::DMatrix::from_row_slice(10,10,
            &vec![ -1., 1., 1., 1., 1., -1., 1., 1., 1., 1.,
                   1., -1., 1., 1., 1., 1., -1., 1., 1., 1., 
                   1., 1., -1., 1., 1., 1., 1., -1., 1., 1., 
                   1., 1., 1., -1., 1., 1., 1., 1., -1., 1., 
                   1., 1., 1., 1., -1., 1., 1., 1., 1., -1., 
                   1., -1., -1., -1., -1., -1., 1., 1., 1., 1., 
                   -1., 1., -1., -1., -1., 1., -1., 1., 1., 1., 
                   -1., -1., 1., -1., -1., 1., 1., -1., 1., 1., 
                   -1., -1., -1., 1., -1., 1., 1., 1., -1., 1., 
                   -1., -1., -1., -1., 1., 1., 1., 1., 1., -1., 
            ]), extra_cols: 0,
        },
        BaseMatrix { basemtx: na::DMatrix::from_row_slice(12,12,
            &vec![ 1., 1., 1., 1., -1., -1., -1., 1., 1., 1., -1., -1., 
                   1., 1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 
                   1., 1., 1., -1., -1., 1., 1., 1., -1., -1., -1., 1., 
                   1., -1., -1., 1., -1., -1., 1., -1., -1., -1., -1., -1., 
                   -1., 1., -1., -1., 1., -1., -1., 1., -1., -1., -1., -1., 
                   -1., -1., 1., -1., -1., 1., -1., -1., 1., -1., -1., -1., 
                   -1., 1., 1., 1., -1., -1., -1., -1., -1., -1., 1., 1., 
                   1., -1., 1., -1., 1., -1., -1., -1., -1., 1., -1., 1., 
                   1., 1., -1., -1., -1., 1., -1., -1., -1., 1., 1., -1.,
                   1., -1., -1., -1., -1., -1., -1., 1., 1., -1., 1., 1., 
                   -1., 1., -1., -1., -1., -1., 1., -1., 1., 1., -1., 1., 
                   -1., -1., 1., -1., -1., -1., 1., 1., -1., 1., 1., -1., 
            ]), extra_cols: 0,
        },
    ];

    // Setup eligible matrices.
    let mut eligible: Vec<BaseMatrix> = Vec::new();
    for &(n, k) in dims.iter() {
        for ref bmtx in xmats.iter() {
            if bmtx.basemtx.nrows() == n {
                let mut bmtx = (*bmtx).clone();
                bmtx.extra_cols = k - bmtx.basemtx.ncols();
                eligible.push(bmtx);
            }
        }
    }

    let mut rng = rand::thread_rng();
    let basemtx = rng.choose(&eligible).unwrap();
    let x = basemtx.fill();
    x
} //@}

pub fn y_a_from_x(x: &na::DMatrix<f64>, m_rx: usize, complex: bool) 
    -> (na::DMatrix<f64>, na::DMatrix<f64>) { //{@
    let n_tx = x.nrows();
    let k = x.ncols();

    // Generate random Gaussian matrix A.
    let a = if complex {
        assert!( n_tx % 2 == 0 && m_rx % 2 == 0 );
        rand_complex_matrix(n_tx, m_rx) 
    } else {
        rand_matrix(m_rx,n_tx)
    };

    // Compute Y = A * X
    let mut y = na::DMatrix::from_column_slice(m_rx, k, &vec![0.0; m_rx*k]);
    a.mul_to(&x, &mut y);

    (a, y)
} //@}

pub fn rank_reduce( y: &na::DMatrix<f64>, m: usize )
    -> Option<na::DMatrix<f64>> {
    let (n,_k) = y.shape();
    assert!( n >= m );

    let svd = na::SVD::new(y.clone(), true, false);

    if n != m {
        let u = match svd.u {
            Some(u) => u,
            None => return None,
        };
        let u = u.remove_rows(m, n - m);
        Some( u * y.clone() )
    } else {
        Some( y.clone() )
    }
}

#[derive(Clone)]
struct BaseMatrix { //{@
    basemtx: na::DMatrix<f64>,
    extra_cols: usize,
} //@}
impl BaseMatrix { //{@
    //{@
    /// Fill in extra_cols with iid \pm1 entries.
    //@}
    fn fill(&self) -> na::DMatrix<f64> {
        // Insert the columns.
        let mut mtx = self.basemtx.clone().insert_columns(self.basemtx.ncols(),
                self.extra_cols, 1.0);

        // Flip to -1.0 wp 0.5.
        let mut rng = rand::thread_rng();
        for j in mtx.ncols() - self.extra_cols .. mtx.ncols() {
            for i in 0 .. mtx.nrows() {
                if rng.gen_range(0, 2) == 1 {
                    mtx.column_mut(j)[i] = -1.0;
                }
            }
        }
        mtx
    }
} //@}

#[allow(dead_code)]
//{@
/// Return generator for all ATM's of a given matrix.
//@}
fn all_atm(mtx: &na::DMatrix<f64>) -> Box<FnMut() -> Option<na::DMatrix<f64>>> { //{@
    let mtx = mtx.clone();
    let n = mtx.shape().0;
    let perms = gen_perm(n);
    let mut perm_idx = 0;
    let mut sign_idx = 0;

    // Generating closure for all n! * 2^n ATM's of mtx.  Outer iteration is
    // over all permutations, which are pre-generated in perms.  Inner iteration
    // is over all sign matrices foreach permutation.
    Box::new(move || {
        // If all permutations already exhausted, return None.
        if perm_idx == perms.len() {
            return None;
        }

        // Get the current permutation, create a clean matrix.
        let perm = &perms[perm_idx];
        let mut next_mtx = mtx.clone();

        // For every row, do 2 things:
        //     1) If it should in fact be permuted, get the right row from mtx.
        //     2) If it should be sign flipped, do that.
        for i in 0 .. n {
            // Right row of permutation.
            if perm[i] != i {
                for j in 0 .. n {
                    next_mtx.row_mut(i)[j] = mtx.row(perm[i])[j];
                }
            }
            // Sign flip.
            if sign_idx & (1 << i) != 0 {
                for j in 0 .. n {
                    next_mtx.row_mut(i)[j] *= -1.0;
                }
            }
        }

        // Increment inner loop over sign matrices; if done increment outer loop
        // over permutation matrices.
        sign_idx += 1;
        if sign_idx == (1 << n) {
            sign_idx = 0;
            perm_idx += 1;
        }

        // Return the matrix produced during this iteration.
        Some(next_mtx)
    })
} //@}
//{@
/// Create all permutations of {0, 1, ..., n-1}.
//@}
fn gen_perm(n: usize) -> Vec<Vec<usize>> { //{@
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![ vec![0] ];
    }

    let mut ret = Vec::new();
    let lower = gen_perm(n - 1);
    for base in lower {
        for i in 0 .. base.len() + 1 {
            let mut cur = base.clone();
            cur.insert(i, n - 1);
            ret.push(cur);
        }
    }
    ret
} //@}

pub fn to_parsable_vector( m: &na::DMatrix<f64> ) 
    -> String {
    let mut string = String::new();
    
    string += "\t[";

    let mut ctr = 0;
    for elt in m.iter() {
        string += &format!(" {},", elt);
        ctr += 1;
        if ctr % 4 == 0 {
            string += "\n\t";
        }
    }
    string += "]";
    string
}

///
/// Updates A_inv in place according to the Sherman-Morrison formula:
/// (A + uv^T)-1 = A^-1 - (A^1 uv^T A^-1) / (1 + v^T A^-1 u)
/// u is just an element of e_j; j is specified as urow
/// Based on benchmarks, this only seems faster starting at n=5
///    
pub fn update_inverse(ainv: &mut na::DMatrix<f64>,  
                      urow: usize, v: &na::DMatrix<f64>) {
    //(A + uv^T)-1 = A^-1 -
    //(A^1 uv^T A^-1) / (1 + v^T A^-1 u)
    
    let ac = ainv.clone();

    let ainv_u = ac.column(urow);
    let mut scale = v.transpose() * ainv_u;
    scale[(0,0)] += 1.0;
    let mut p2 = v.transpose() * ainv.clone(); 
    p2 /= scale[(0,0)];

    *ainv -= ainv_u * p2;
}

///
/// Similar to above but takes in the transpose of the inverse
///
pub fn update_inverse_transpose(ainv_trans: &mut na::DMatrix<f64>,  
                                urow: usize, v: &na::DMatrix<f64>) {
    //(A + uv^T)-1 = A^-1 -
    //(A^1 uv^T A^-1) / (1 + v^T A^-1 u)
    
    let ac = ainv_trans.transpose().clone();

    let ainv_u = ac.column(urow);
    let mut scale = v.transpose() * ainv_u;
    scale[(0,0)] += 1.0;
    let mut p2 = v.transpose() * ainv_trans.clone().transpose(); 
    p2 /= scale[(0,0)];

    *ainv_trans -= p2.transpose() * ainv_u.transpose();
}

///
/// Uses matrix determinant lemma to find change in ln |det a| when delta (a row vector)
/// is added to row corresponding to 'row' of 'a'.  Returns -infty if result is singular
///
/// log|det (A + e_i * Delta)| = log|(1 + Delta A^-1 e_i)| + log|det(U)|
///
/// This is faster starting at n=4
///
pub fn delta_log_det(ainv: &na::DMatrix<f64>, delta: &na::DMatrix<f64>,
                     row: usize) -> f64 {
    let ac = ainv.clone();
    let ainv_col = ac.column(row);
    let prod = delta * ainv_col;

    let res = (prod[(0,0)] + 1.0).abs();

    if res > 0.0 {
        return res.ln();
    } else {
        return std::f64::NEG_INFINITY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use self::test::Bencher;

    #[test]
    fn update_inverse_test() {
        for _ in 0 .. 100 {
            let n = 5;
            let mut rng = rand::thread_rng();
            let urow = rng.gen_range(0, n);
            let a = rand_matrix( n, n );
            let v = rand_matrix( n, 1 );       
            let mut u = na::DMatrix::from_column_slice( n, 1, &vec![0.0; n] );
            u[(urow, 0)] = 1.0;

            //Compute using update formula
            let mut ainv = a.clone().try_inverse().unwrap();
            update_inverse( &mut ainv, urow, &v );

            //Compute update directly
            let a_update = a.clone() + u.clone() * v.transpose();
            let inv2 = a_update.clone().try_inverse().unwrap();

            let error = ainv - inv2;
            assert!( 
                error.iter().fold( 0.0, |acc, &e| acc + e.abs())
                < 1e-6 
            );
        }
    }

    #[test]
    fn update_log_det_test() {
        for _ in 0 .. 100 {
            let n = 5;
            let mut rng = rand::thread_rng();
            let urow = rng.gen_range(0, n);
            let mut a = rand_matrix( n, n );
            let v = rand_matrix( n, 1 );       
            let mut u = na::DMatrix::from_column_slice( n, 1, &vec![0.0; n] );
            u[(urow, 0)] = 1.0;

            //Update the fast way
            let ln_det = a.determinant().abs().ln();
            let ainv = a.clone().try_inverse().unwrap();
            let delta = delta_log_det( &ainv, &v.transpose(), urow );

            //Compute from scratch
            a = a.clone() + u.clone() * v.transpose();
            let ln_det_update = a.clone().determinant().abs().ln();

            assert!( (ln_det_update - ln_det - delta).abs() < 1e-6 );
        }
    }

    #[bench]
    fn bench_update_inverse(b: &mut Bencher) {
        let n = 5;
        let mut rng = rand::thread_rng();
        let urow = rng.gen_range(0, n);
        let a = rand_matrix( n, n );
        let v = rand_matrix( n, 1 );
        let mut ainv = a.try_inverse().unwrap();

        b.iter(|| update_inverse( &mut ainv, urow, &v ) );
    }

    #[bench]
    fn bench_direct_inverse(b: &mut Bencher) {
        let n = 5;
        let a = rand_matrix( n, n );

        b.iter(|| a.clone().try_inverse() );
    }

    #[bench]
    fn bench_update_log_det(b: &mut Bencher) {
        let n = 8;
        let mut rng = rand::thread_rng();
        let urow = rng.gen_range(0, n);
        let a = rand_matrix( n, n );
        let v = rand_matrix( n, 1 );       

        let ainv = a.clone().try_inverse().unwrap();
        
        b.iter( || delta_log_det( &ainv, &v.transpose(), urow ) );       
    }

    #[bench]
    fn bench_direct_log_det(b: &mut Bencher) {
        let n = 8;
        let a = rand_matrix( n, n );
        
        b.iter( || a.determinant().abs().ln() );       
    }
}
