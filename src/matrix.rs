extern crate nalgebra as na;
extern crate rand;

use rand::distributions::{Normal, Distribution};
use rand::Rng;

use is_feasible;

#[allow(dead_code)]
pub fn rand_unit(n: usize) -> na::DMatrix<f64> {
    let mut v = rand_matrix( 1, n );
    let norm = v.norm();
    v /= norm;

    return v;
}

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
pub fn rand_complex_matrix( n: usize ) -> na::DMatrix<f64> {
    let nb = n /2;

    let mut rng = rand::thread_rng();
    let mut real_data = Vec::with_capacity(nb * nb);
    let mut imag_data = Vec::with_capacity(nb * nb);
    let mut data = Vec::with_capacity(n * n);

    let dist = Normal::new(0.0, 1.0);
    for _ in 0 .. (nb * nb) {
        real_data.push(dist.sample(&mut rng));
        imag_data.push(dist.sample(&mut rng));
    }

    for i in 0 .. nb {
        data.extend(&real_data[(nb*i) .. nb*(i+1)]);
        data.extend(&imag_data[(nb*i) .. nb*(i+1)]);
    }

    for i in 0 .. nb {
        //v = -1.0 * imag_data[nb*i .. nb*(i+1)]
        let v = imag_data
            .iter()
            .cloned()
            .map(|x| -1.0 * x)
            .skip(nb*i)
            .take(nb)
            .collect::<Vec<f64>>();

        data.extend(&v);

        data.extend(&real_data[(nb*i) .. nb*(i+1)]);
    }

    na::DMatrix::from_row_slice(n, n, &data)
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
