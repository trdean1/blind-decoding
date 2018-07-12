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
