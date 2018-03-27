// imports{@
#![allow(unused_imports)]
extern crate nalgebra as na;
extern crate rand;
extern crate regex;
#[macro_use]
extern crate log;
extern crate env_logger;
use rand::distributions::IndependentSample;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use regex::Regex;
use std::cmp::Ordering;
use rand::Rng;
use std::collections::{HashSet, BTreeSet, HashMap};
use std::thread;
use std::sync::{Arc, Mutex};
use std::fmt;
use std::error;
use std::error::Error;
use std::process;
// end imports@}
// constants{@
#[allow(unused)]
const VERBOSE_HOP: u64 = 0x1;
#[allow(unused)]
const VERBOSE_BFS: u64 = 0x2;
#[allow(unused)]
const VERBOSE_CONSTR_CREATE: u64 = 0x4;
#[allow(unused)]
const VERBOSE_TAB_MAP: u64 = 0x8;
#[allow(unused)]
const VERBOSE_EXTRA_CONSTR: u64 = 0x10;
#[allow(unused)]
const VERBOSE_EVAL_VERTEX: u64 = 0x20;
#[allow(unused)]
const VERBOSE_FLIP_GRADIENT: u64 = 0x40;
#[allow(unused)]
const VERBOSE_FLIP: u64 = 0x80;
#[allow(unused)]
const VERBOSE_INITIAL_TAB: u64 = 0x100;
#[allow(unused)]
const VERBOSE_INDEP: u64 = 0x10000;
#[allow(unused)]
const VERBOSE_VERTEX: u64 = 0x20000;
#[allow(unused)]
const VERBOSE_ALL: u64 = 0xFFFFFFFF;

const ZTHRESH: f64 = 1e-9;
const VERBOSE: u64 = 0;
// end constants@}

fn main() { //{@
    // Initialize logger.
    env_logger::init();

    // Enumeration functions. 
    //neighbor_det_pattern("./n6_equiv.txt");
    //neighbor_det_pattern("./n8_equiv.txt");
    //neighbor_det_pattern_all(5);

    // Run repetitions of the solver.
    //run_reps();
    
    test_awgn();
} //@}

// Principal functions{@
// Runnable functions{@
#[derive(Hash, Eq, PartialEq)]
enum BFSType { //{@
    PM1,
    PM10,
    Wrong,
} //@}
#[allow(dead_code)]//{@
/// Run the BFS finder multiple times over multiple different base matrices.
//@}
fn many_bfs(reps: usize) { //{@
    let xmats = vec![
        BaseMatrix { basemtx: na::DMatrix::from_row_slice(2, 2,
                &vec![  1.,  1.,
                        1., -1., 
                ]), extra_cols: 3,
        },
        BaseMatrix { basemtx: na::DMatrix::from_row_slice(3, 4,
                &vec![  1.,  1.,  1.,  1.,
                        1.,  1., -1., -1.,
                        1., -1.,  1., -1.,
                ]), extra_cols: 5,
        },
        BaseMatrix { basemtx: na::DMatrix::from_row_slice(4, 5,
                &vec![  1.,  1.,  1.,  1.,  1.,
                        1.,  1., -1., -1.,  1.,
                        1., -1.,  1., -1.,  1.,
                        1., -1., -1.,  1., -1.,
                ]), extra_cols: 5,
        },
        BaseMatrix { basemtx: na::DMatrix::from_row_slice(6, 6,
                &vec![ -1.,  1.,  1.,  1.,  1.,  1.,
                        1., -1.,  1.,  1.,  1.,  1.,
                        1.,  1., -1.,  1.,  1.,  1.,
                       -1., -1., -1., -1.,  1.,  1.,
                       -1., -1., -1.,  1., -1.,  1.,
                       -1., -1., -1.,  1.,  1., -1.,
                ]), extra_cols: 8,
        },
    ];
    let mut rng = rand::thread_rng();
    let mut results = HashMap::new();
    for _ in 0 .. reps {
        let basemtx = rng.choose(&xmats).unwrap();
        let x = basemtx.fill();
        let (a, y) = trial(&x, false);
        let u_i = rand_init(&y);
        //println!("a = {}\ny = {}\nUi = {}", a, y, u_i);
        let bfs = find_bfs(&u_i, &y);
        let res = verify_bfs(&bfs, &y, ZTHRESH);
        if res == BFSType::Wrong {
            let uy = bfs.clone() * y.clone();
            println!("BFS Fail: a = {:.6}y = {:.6}u_i = {:.6}u = {:.6}\nuy = {:.6}",
                    a, y, u_i, bfs, uy);
        }
        let count = results.entry(res).or_insert(0);
        *count += 1;
    }

    for (res, count) in &results {
        println!("{:-10}{}", match res {
            &BFSType::PM1 => "+-1",
            &BFSType::PM10 => "+-1/0",
            &BFSType::Wrong => "WRONG",
            },
            count);
    }
} //@}
#[allow(dead_code)]
//{@
/// Return set matrices: (y, u_i) used for debugging purposes.
//@}
fn use_given_matrices() -> (na::DMatrix<f64>, na::DMatrix<f64>) { //{@
    let y = na::DMatrix::from_row_slice(3, 8,
            &vec![
            -2.13221971, 1.65267722, -3.58171708, 0.20317985,
            -0.20317985, -1.65267722, -2.13221971, 2.13221971,

            1.33427445, 0.75515798, -1.76382965, -2.34294612, 
            2.34294612, -0.75515798, 1.33427445, -1.33427445,

            -2.49940914, 0.07649133, -0.74610697, 1.8297935,
            -1.8297935, -0.07649133, -2.49940914, 2.49940914,
            ]);

    let u_i = na::DMatrix::from_row_slice(3, 3,
            &vec![
            -0.04251098, -0.1185273, 0.21597244,
            0.22137548, 0.07778901, 0.08626569,
            0.10810049, -0.20591296, -0.0917286,
            ]);
    
    (y, u_i)
} //@}

#[allow(dead_code)]
fn near_pm_one( x: f64, tol: f64 ) -> f64 {
    if ( x - 1f64 ).abs() < tol {
        return x - 1f64;
    } else if ( x + 1f64 ).abs() < tol {
        return x + 1f64;
    }

    0f64
}

fn center_y( u: na::DMatrix<f64>, y: na::DMatrix<f64>, tol: f64 ) -> 
    Option<na::DMatrix<f64>> 
{
    //Skip if tolerance is zero (i.e. don't center)
    if tol == 0f64 {
        return Some(y);
    }

    let uy = u.clone() * y.clone();

    //Find epsilon
    let del = uy.map(|x| near_pm_one(x, tol) );

    //center y, first we need bfs_inv
    let qr = na::QR::new( u.clone() );
    let bfs_inv_o = qr.try_inverse();
    match bfs_inv_o {
        Some(u_inv) => return Some(y - u_inv * del),
        None => return None,
    }
}


#[allow(dead_code)]
/// Testing AWGN performance
fn test_awgn() {
    let reps_per = 10;

    let n = 4; let k = 10;
    let complex = false;
    let var = 0.0001; 
    let tol = 0.03;
    
    let dim = vec![(n, k)];

    //Generate trial and add noise
    let x = get_matrix( &dim[0 .. 1] );
    let (_a, y_base) = trial(&x, complex);

    for _ in 0 .. reps_per {
        let e = rand_matrix(n, k);
        let mut y = y_base.clone() + var*e;

        match single_run(&y,true,tol) {
            Err(e) => {
                println!("Some sort of error: {}", e);
            },
            Ok(ft) => {
                println!("Worked!");
                let ref best_state = ft.best_state;
                println!("{:.3}", best_state.uy);
            }
        };
    }
}

#[allow(dead_code)]
//{@
/// Run repetitions of multiple different matrix dimensions, recording success /
/// failure for each matrix types.
//@}
fn run_reps() { //{@
    let complex = false;
    let use_basis = true; 
    let reps_per = 1000;
    //let dims = vec![(2, 3), (3, 6), (4, 6), (4, 8), (5, 9), (5, 12), (5, 15),
    //    (6, 12), (6, 18), (6, 24), (8, 20), (8, 28), (8, 36)];

    // DEBUG: use this in addition to splice in get_matrix to use static X.
    //let dims = vec![(4, 6)];
    
    let mut dims = Vec::new();
    for ii in 2 .. 9 {
        if ii == 7 { continue; }
        for jj in 0 .. 16 {
            if 2*jj <= ii { continue; }
            dims.push( (ii, 2*jj) );
        }
    }
    
    // Setup basic results vector: should migrate this to a full structure.
    // { (n, k), attempts, success, fail = \pm1, fail != \pm1, duration }, 
    let mut results: Vec<((usize, usize), u64, u64, u64, u64, u64, f64)> = dims.iter()
        .map(|&d| (d, 0, 0, 0, 0, 0, 0.0))
        .collect::<Vec<_>>();

    for _iter in 0 .. reps_per * dims.len() {
        let which = _iter / reps_per;
        
        if complex && (dims[which].0 % 2 != 0) {
            warn!("Complex case must have even n");
            continue;
        }

        if _iter % reps_per == 0 {
            eprintln!("Dim = {:?}", dims[which]);
        } else if _iter % 100 == 0 {
            eprint!("#");
        } else if _iter % reps_per == reps_per - 1 {
            eprint!("\n");
        }

        // Select X matrix of one specific set of dimensions (n, k).
        let x = if use_basis {
            get_matrix(&dims[which .. which + 1]) 
        } else {
            rand_pm1_matrix(dims[which].0, dims[which].1)
        };

        trace!("selected x = {}", x);
        // Get pointer to relevant results tuple for this dimension.
        let ref mut res = results.iter_mut().find(|ref e| e.0 == x.shape()).unwrap();
        res.1 += 1;

        // Obtain A, Y matrices, then run.
        let (a, y) = trial(&x, complex);
        let timer = std::time::Instant::now();
        match single_run(&y,use_basis, 0f64) {
            Err(e) => {
                match e {
                    FlexTabError::Runout => {
                        res.5 += 1; // ran out
                        debug!("ran out of attempts");
                    },
                    _ => {
                        res.4 += 1; // something else -- problem
                        println!("critical error = {}", e);
                    },
                };
            },
            Ok(ft) => {
                // Obtained a result: check if UY = X up to an ATM.
                let ref best_state = ft.best_state;
                if equal_atm(&best_state.uy, &x) {
                    debug!("EQUAL ATM");
                    res.2 += 1;
                } else {
                    // UY did +not+ match X, print some results and also
                    // determine if UY was even a vertex.
                    match a.try_inverse() {
                        None => debug!("UNEQUAL: Cannot take a^-1"),
                        Some(inv) => debug!("UNEQUAL: u = {:.3}a^-1 = {:.3}", 
                                best_state.u, inv),
                    };
                    debug!("UNEXPECTED: return non-ATM");
                    debug!("uy = {:.3}", best_state.uy);
                    if best_state.uy.iter().all(|&e|
                            (e.abs() - 1.0).abs() < ft.zthresh) {
                        res.3 += 1; // UY = \pm 1
                    } else {
                        res.4 += 1; // UY != \pm 1 -- problem
                        println!("critical error: uy = {:.3}", best_state.uy);
                    }
                }
            },
        };
        // Charge elapsed time for this run to its (n, k) dimension.
        let elapsed = timer.elapsed();
        res.6 += elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 * 1e-9;
    }

    // Print overall results.
    for ref res in results.iter() {
        let mut output = 
            format!("n = {}, k = {:2}: success = {:4} / {:4}, ",
                    (res.0).0, (res.0).1, res.2, res.1);
        output += &format!("(runout = {:2}, pm1 = {:2} err = {:2}), ",
                res.5, res.3, res.4);
        output += &format!("mean_time_per = {:.5e}", res.6 / res.1 as f64);
        println!("{}", output);
    }
} //@}
//{@
/// Perform a single run using a given +y+ matrix, which contains k symbols each
/// of length n.
//@}
fn single_run(y: &na::DMatrix<f64>, skip_check: bool, center_tol: f64) -> Result<FlexTab, FlexTabError> { //{@
    let mut attempts = 0;
    const LIMIT: usize = 100; // Max attempts to restart with fresh random U_i.
    let mut ft;
    let mut best: Option<FlexTab> = None;

    //If true, check that y is not singular
    if skip_check == false {
       let (n,_k) = y.shape();
       let svd = na::SVD::new(y.clone(), false, false);
       let s = svd.singular_values;
       trace!("Singular values =\n");
       for ss in s.iter() {
           trace!("{}", ss);
       }
       let r = s.iter().filter(|&elt| *elt > ZTHRESH).count();
       trace!("({})\n",r);
       if r < n {
            return Err(FlexTabError::SingularInput);
       }
    }


    // Loop trying new u_i until we get n linearly independent \pm 1 cols.
    loop {
        attempts += 1;
        if attempts > LIMIT {
            match best {
                Some(b) => return Ok(b),
                None => return Err(FlexTabError::Runout),
            };
        }
        let u_i = rand_init(&y); // Choose rand init start pt.
        //let (y, u_i) = use_given_matrices(); // Use for debugging only.

        info!("y = {:.8}Ui = {:.8}", y, u_i);
        let bfs = find_bfs(&u_i, &y); // Find BFS.
        info!("bfs = {:.8}", bfs);
        info!("uy = {:.8}", bfs.clone() * y.clone() );

        //TODO: This is horrible code and needs to get cleaned up
        //Center and then create flex tab
        //couldn't figure out how to change reference...
        match center_y( bfs.clone(), y.clone(), center_tol ) {
            Some(y) => {
                 ft = match FlexTab::new(&bfs, &y, ZTHRESH) {
                    Ok(ft) => ft,
                    Err(e) => match e {
                        // Insufficient good cols => retry.
                        FlexTabError::GoodCols => {
                            debug!("Insufficient good cols, retrying...");
                            continue;
                        },
                        // Any other error => propogate up.
                        _ => return Err(e),
                    },
                };
            },
            None => {
                info!("Error centering, dropping attempt");         
                ft = match FlexTab::new(&bfs, &y, ZTHRESH) {
                    Ok(ft) => ft,
                    Err(e) => match e {
                        // Insufficient good cols => retry.
                        FlexTabError::GoodCols => {
                            debug!("Insufficient good cols, retrying...");
                            continue;
                        },
                        // Any other error => propogate up.
                        _ => return Err(e),
                    },
                };
            },
        }
        //TODO: end horrible code

        // Now we have at least n good cols, so try to solve.  If error is that
        // we don't have n linearly independent good cols, then try new u_i.
        // Do same thing if we appear to have been trapped.
        debug!("num_good_cols = {}, initial ft =\n{}", ft.num_good_cols(), ft);
        match ft.solve() {
            Ok(_) => break,
            Err(e) => match e {
                FlexTabError::LinIndep => debug!("LinIndep, retrying..."),
                FlexTabError::StateStackExhausted | FlexTabError::TooManyHops
                    => {
                        best = match best {
                            Some(b) => if ft.state.obj > b.state.obj
                                { Some(ft) } else { Some(b) },
                            None => Some(ft),
                        };
                        debug!("{}, retrying...", e);
                    },
                _ => return Err(e),
            },
        }
    }

    debug!("FlexTab n = {}: {}, visited = {}", ft.n, 
            if ft.ybad.is_some() { "REDUCED" } else { "FULL" }, ft.visited.len());

    // If FlexTab is reduced, we need to do this again starting with a real BFS.
    // Here there is no possibility of insufficient good cols or lin indep cols.
    match ft.ybad {
        Some(_) => {
            debug!("Reduced: now need to solve...");
            let mut ftfull = FlexTab::new(&ft.state.u.clone(), &y, ZTHRESH)?;
            match ftfull.solve() {
                Ok(_) => Ok(ftfull),
                Err(e) => {
                    println!("ftfull from reduced err = {}", e);
                    match e {
                        FlexTabError::StateStackExhausted
                            | FlexTabError::TooManyHops
                            => Ok(ftfull),
                        _ => Err(e),
                    }
                }
            }
        },
        None => Ok(ft),
    }
} //@}
// end runnable functions@}

// matrix generation functions{@
#[allow(dead_code)]
//{@
///Generate a random Gaussian(0, 1) matrix of the given dimensions.
//@}
fn rand_matrix(nrows: usize, ncols: usize) -> na::DMatrix<f64> { //{@
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(nrows * ncols);
    let dist = rand::distributions::Normal::new(0.0, 1.0);
    for _ in 0 .. (nrows * ncols) {
        data.push(dist.ind_sample(&mut rng));
    }
    na::DMatrix::from_column_slice(nrows, ncols, &data)
} //@}

///Genreate a random Gaussian(0, 1) complex matrix that is n x n.
///represented as [re im; -im re]
#[allow(dead_code)]
fn rand_complex_matrix( n: usize ) -> na::DMatrix<f64> {
    let nb = n /2;

    let mut rng = rand::thread_rng();
    let mut real_data = Vec::with_capacity(nb * nb);
    let mut imag_data = Vec::with_capacity(nb * nb);
    let mut data = Vec::with_capacity(n * n);

    let dist = rand::distributions::Normal::new(0.0, 1.0);
    for _ in 0 .. (nb * nb) {
        real_data.push(dist.ind_sample(&mut rng));
        imag_data.push(dist.ind_sample(&mut rng));
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


#[allow(dead_code)]
//{@
/// Generate a random \pm 1 matrix of the given dimensions.
//@}
fn rand_pm1_matrix(nrows: usize, ncols: usize) -> na::DMatrix<f64> { //{@
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(nrows * ncols);
    for _ in 0 .. (nrows * ncols) {
        data.push(1.0 - (2 * rng.gen_range(0, 2)) as f64);
    }
    na::DMatrix::from_column_slice(nrows, ncols, &data)
} //@}
#[allow(dead_code)]
//{@
/// Return closure that generates all matrices in {-1, +1} ^ {n x n}.
//@}
fn gen_all_pm1(n: usize) -> Box<FnMut() -> Option<na::DMatrix<f64>>> { //{@
    let mut count = 0;
    Box::new(move || {
        if count >= (1 << n*n) { return None; }
        // Get next matrix.
        let mut data = Vec::with_capacity(n * n);
        for i in 0 .. n*n {
            let e = if count & (1 << i) != 0 { 1.0 } else { -1.0 };
            data.push(e);
        }
        count += 1;
        Some(na::DMatrix::from_row_slice(n, n, &data))
    })
} //@}
//{@
/// Return true iff a == b up to an ATM.
//@}
fn equal_atm(a: &na::DMatrix<f64>, b: &na::DMatrix<f64>) -> bool { //{@
    if a.shape() != b.shape() { return false; }
    // used_rows[k] will be set to true when a row k of +b+ is used to match
    // some row in +a+.
    let mut used_rows = vec![false; b.shape().0];

    // Iterate over all rows of +a+; foreach row find a (previously unmatched)
    // row of +b+ that matches this row of +a+ up to a sign inversion.
    for i in 0 .. a.shape().0 {
        let mut matched = false;
        for k in 0 .. used_rows.len() {
            if used_rows[k] { continue; }
            // Calc difference of vector a[i] - b[i].
            if (a.row(i) - b.row(k)).amax() < ZTHRESH
                    || (a.row(i) + b.row(k)).amax() < ZTHRESH {
                matched = true;
                used_rows[k] = true;
            }
        }
        if !matched { return false; }
    }
    true
} //@}
//{@
/// Randomly select one (n, k) value from the array of (n, k) pairs given in
/// +dims+.  Fill the necessary extra columns (if necessary) with random iid
/// \pm 1 entries.
//@}
fn get_matrix(dims: &[(usize, usize)]) -> na::DMatrix<f64> { //{@
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

    // DEBUG: start splice for specific X
    /*
    let xmats = vec![
        BaseMatrix { basemtx: na::DMatrix::from_row_slice(4, 6,
                &vec![
                -1.0, -1.0, -1.0, -1.0,  1.0, -1.0,
                1.0, -1.0, -1.0,  1.0, -1.0,  1.0,
                1.0, -1.0,  1.0, -1.0, -1.0, -1.0,
                -1.0, -1.0, -1.0,  1.0,  1.0, -1.0,
                ]), extra_cols: 0,
        },
    ];
    */
    // DEBUG: end splice for specific X

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
// end matrix generation functions@}

// BFS finder functions.{@
//{@
/// Generator for all pairs of items from a HashSet.
//@}
fn gen_pairs<T: 'static>(h: &HashSet<T>) -> Box<FnMut() -> Option<(T, T)>> //{@
        where T: Copy + Eq + std::hash::Hash {
    let vec = h.iter().map(|&e| e).collect::<Vec<_>>();
    let mut i = 0;
    let mut j = 1;

    Box::new(move || {
        if i >= vec.len() || j >= vec.len() { return None; }
        let ret = (vec[i], vec[j]);
        j += 1;
        if j >= vec.len() {
            i += 1;
            j = i + 1;
        }
        Some(ret)
    })
} //@}
//{@
/// Generate matrix a of same dimension as _x_, each entry iid ~N(0, 1).
/// Generate matrix y = a * x.
/// Return (a, y)
//@}
fn trial(x: &na::DMatrix<f64>, complex: bool) -> (na::DMatrix<f64>, na::DMatrix<f64>) { //{@
    trace!("X = {}", x);
    let n = x.nrows();
    let k = x.ncols();

    // Generate random Gaussian matrix A.
    let a = if complex {
        rand_complex_matrix(n) 
    } else {
        rand_matrix(n,n)
    };

    // Compute Y = A * X
    let mut y = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
    a.mul_to(&x, &mut y);

    (a, y)
} //@}
//{@
/// Calculate gradient: (x^-1).transpose.
//@}
fn objgrad(x: &mut na::DMatrix<f64>) -> Option<na::DMatrix<f64>> { //{@
    match x.try_inverse_mut() {
        true => Some(x.transpose()),
        false => None,
    }
} //@}
//{@
/// Generate a random orthogonal n x n matrix.
//@}
fn random_orthogonal(n: usize) -> na::DMatrix<f64> { //{@
    na::QR::new(rand_matrix(n, n)).q()
} //@}
//{@
/// Find a random feasible point, which is an n x n U such that all entries of
/// UY are bounded by 1 in absolute value.
/// Input:   Y = n x k matrix of received symbols.
/// Output:  U = n x n feasible inverse of the channel gain matrix.
//@}
fn rand_init(y: &na::DMatrix<f64>) -> na::DMatrix<f64> { //{@
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
/// Returns true if the given value of U is feasible.  Ignores entries in the
/// product UY where mask is set to 0.  This is done to ignore entries that
/// correspond to active constraints.
//@}
fn is_feasible(u: &na::DMatrix<f64>, y: &na::DMatrix<f64>, //{@
        mask: Option<&na::DMatrix<bool>>) -> bool {
    let prod = u * y;
    if let Some(mask) = mask {
        assert!(prod.shape() == mask.shape());
    }

    for j in 0 .. prod.ncols() {
        for i in 0 .. prod.nrows() {
            let check = match mask {
                None => true,
                Some(mask) => mask.column(j)[i],
            };
            if check && prod.column(j)[i].abs() > 1.0f64 + ZTHRESH {
                return false;
            }
        }
    }
    true
} //@}
//{@
/// Returns a boolean matrix where each entry tells whether or not the
/// constraint is active.
/// Input:   u = feasible solution (n x n)
///          y = matrix of received symbols (n x k)
/// Output:  Boolean matrix (n x k) where output[i, j] = true iff the ith row of
/// u causes the jth constraint of UY to be active, |<u_i, y_j>| = 1.        
//@}
fn get_active_constraints_bool(u: &na::DMatrix<f64>, //{@
        y: &na::DMatrix<f64>, c_bool: &mut na::DMatrix<bool>) {
    let prod = u * y;
    let (n, k) = prod.shape();

    for j in 0 .. k {
        let mut c_bool_col = c_bool.column_mut(j);
        let prod_col = prod.column(j);
        for i in 0 .. n {
            let entry = prod_col[i];
            c_bool_col[i] =  (1.0 - entry).abs() < ZTHRESH 
                || (1.0 + entry).abs() < ZTHRESH;
        }
    }
} //@}
//{@
/// For each entry in +update+, append the corresponding row to +p+.
//@}
fn update_p(mut p: na::DMatrix<f64>, update: &Vec<(usize, usize)>, //{@
        y: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let (n, _k) = y.shape();
    for &(i, j) in update.iter() {
        // Add the new row, copy in transpose of normalized column.
        let row_idx = p.nrows();
        p = p.insert_row(row_idx, 0.0);
        let ycol = y.column(j);
        let mut new_row = p.row_mut(row_idx);
        let norm = ycol.norm();
        let new_row_baseidx = i * n;
        for q in 0 .. n {
            new_row[new_row_baseidx + q] = ycol[q] / norm;
        }
    }
    p
} //@}
//{@
/// Convert p into an orthogonal basis via de-facto optimized QR decomposition.
//@}
fn orthogonalize_p(mut p: na::DMatrix<f64>) -> na::DMatrix<f64> { //{@
    let (c, nsq) = p.shape();
    let pt = p.clone().transpose();
    let mut a = na::DMatrix::from_column_slice(c, c, &vec![0.0; c*c]);
    p.mul_to(&pt, &mut a);

    let mut bad = HashSet::new();
    for j in 0 .. c {
        let col = a.column(j);
        for i in (j+1) .. c {
            if col[i].abs() > ZTHRESH {
                bad.insert(i);
                bad.insert(j);
            }
        }
    }
    debug!("a = {:.3}", a);
    debug!("Non-orthogonal vectors: {:?}", bad);

    let mut zero = HashSet::new();
    let mut pair_generator = gen_pairs(&bad);
    let mut u = na::DMatrix::from_column_slice(1, nsq, &vec![0.0; nsq]);
    let mut v = na::DMatrix::from_column_slice(1, nsq, &vec![0.0; nsq]);
    let mut v_t = na::DMatrix::from_column_slice(nsq, 1, &vec![0.0; nsq]);
    let mut u_vt = na::DMatrix::from_column_slice(1, 1, &vec![0.0; 1]);
    let mut vv = na::DMatrix::from_column_slice(1, nsq, &vec![0.0; nsq]);
    while let Some((i, j)) = pair_generator() {
        if zero.contains(&i) || zero.contains(&j) { continue; }
        u.copy_from(&p.row(i));
        v.copy_from(&p.row(j));
        v_t.tr_copy_from(&v);       // v_t = p[j,:].T
        u.mul_to(&v_t, &mut u_vt);  // u_vt = u * v.T
        {
            let u_vt = u_vt.column(0)[0];
            u.apply(|e| e * u_vt);      // u = (u * v.T) * u
        }
        v.sub_to(&u, &mut vv);      // vv = v - ((u * v.T) * u
        let norm_vv = vv.norm();
        if norm_vv < ZTHRESH {
            zero.insert(j);
        } else {
            vv /= norm_vv;
            p.row_mut(j).copy_from(&vv);
        }
    }

    let mut zero = zero.iter().map(|&e| e).collect::<Vec<_>>();
    zero.sort_by(|&a, &b| b.cmp(&a));
    debug!("Redundant constraints: {:?}", zero);
    for &z in zero.iter() {
        p = p.remove_row(z);
    }
    p
} //@}
//{@
/// Calculate the distance to the problem boundary along a given vector.
/// Input:   u = current feasible solution
///          v = vector along which to travel
///          y = received symbols
///          mask = points to consider when calculating feasibility; if None,
///          all points are considered.
/// Output: t = maximum distance such that u + t * v remains feasible.
//@}
fn boundary_dist(u: &na::DMatrix<f64>, v: &na::DMatrix<f64>, //{@
        y: &na::DMatrix<f64>, mask: Option<&na::DMatrix<bool>>) -> f64 {
    let uy = u * y;
    let dy = v * y;
    let mut t_min = std::f64::MAX;

    // Find the lowest value of t such that U + t * V reaches the boundary.
    for j in 0 .. y.shape().1 {
        for i in 0 .. y.shape().0 {
            if let Some(mask) = mask {
                if mask.column(j)[i] { continue; }
            }

            // Determine value of t s.t. [i, j] constr reaches boundary.
            let t = match dy.column(j)[i].partial_cmp(&0.0).unwrap() {
                std::cmp::Ordering::Less =>
                    (-1.0 - uy.column(j)[i]) / dy.column(j)[i],
                std::cmp::Ordering::Greater =>
                    (1.0 - uy.column(j)[i]) / dy.column(j)[i],
                std::cmp::Ordering::Equal => std::f64::MAX,
            };
            if t < t_min { t_min = t; }
        }
    }

    t_min
} //@}
//{@
/// Find an initial BFS for the constraints |UY|\_infty <= 1.  Algorithm works
/// by moving to the problem boundary, then following the projection of the
/// gradient onto the nullspace of active constraints until hittin the boundary
/// again.
/// Input:   u_i = initial estimate for inverse of channel gain matrix
///          y = n x k matrix of received symbols
/// Output:  u = estimated inverse channel gain matrix that forms a BFS
//@}
fn find_bfs(u_i: &na::DMatrix<f64>, y: &na::DMatrix<f64>) -> na::DMatrix<f64> { //{@
    let mut u = u_i.clone();
    let mut gradmtx = u.clone();
    let (n, k) = y.shape();
    let mut v = objgrad(&mut gradmtx).unwrap();
    let mut t = boundary_dist(&u, &v, &y, None);
    v *= t;
    u += v;

    let mut gradmtx = na::DMatrix::from_column_slice(n, n, &vec![0.0; n*n]);
    let mut gradvec = vec![0.0; n*n];
    let mut p_bool = na::DMatrix::from_column_slice(n, k, &vec![false; n*k]);
    let mut p_bool_iter = na::DMatrix::from_column_slice(n, k, &vec![false; n*k]);
    let mut p_bool_updates = Vec::with_capacity(k);
    let mut p = na::DMatrix::from_column_slice(0, n*n, &Vec::<f64>::new());
    for _iter in 0 .. (n*n - 1) {
        if cfg!(build = "debug") {
            let mut _uy = na::DMatrix::from_column_slice(n, y.ncols(),
                    &vec![0.0; n * y.ncols()]);
            u.mul_to(&y, &mut _uy);
            println!("Iteration {}\nuy = {:.3}p = {:.3}", _iter, _uy, p);
        }
        get_active_constraints_bool(&u, &y, &mut p_bool_iter);
        p_bool_updates.clear();
        for j in 0 .. k {
            let col_iter = p_bool_iter.column(j);
            let col_orig = p_bool.column_mut(j);
            for i in 0 .. n {
                if col_iter[i] && !col_orig[i] {
                    p_bool_updates.push((i, j));
                }
            }
        }
        p_bool.copy_from(&p_bool_iter);

        p = update_p(p, &p_bool_updates, &y);
        p = orthogonalize_p(p);

        gradmtx.copy_from(&u);
        gradmtx = match objgrad(&mut gradmtx) {
            Some(grad) => grad,
            None => break,
        };
        gradvec.copy_from_slice(&gradmtx.transpose().as_slice());
        debug!("Iteration {} has {} independent constraints", _iter, p.nrows());
        for row_idx in 0 .. p.nrows() {
            let row = p.row(row_idx);
            let s = gradvec.iter().enumerate().fold(0.0, |sum, (idx, &e)|
                        sum + e * row[idx]);
            gradvec.iter_mut().enumerate().for_each(|(j, e)| *e -= s * row[j]);
        }
        debug!("gradvec = {:?}", gradvec);

        for j in 0 .. n {
            let mut col = gradmtx.column_mut(j);
            for i in 0 .. n {
                let idx = n * i + j;
                col[i] = gradvec[idx];
            }
        }

        if gradmtx.norm() < 1e-12 {
            debug!("Iteration {} gradmtx.norm negligible", _iter);
            break
        }

        t = boundary_dist(&u, &gradmtx, &y, Some(&p_bool));
        gradmtx.apply(|e| e * t);

        //u += gradmtx;
        for j in 0 .. n {
            let mut col_u = u.column_mut(j);
            let col_grad = gradmtx.column(j);
            for i in 0 .. n {
                col_u[i] += col_grad[i];
            }
        }
    }
    u
} //@}
//{@
/// Verify that every entry of |uy| == 1.
//@}
fn verify_bfs(u: &na::DMatrix<f64>, y: &na::DMatrix<f64>, zthresh: f64) //{@
        -> BFSType {
    if u.determinant().abs() < zthresh * 10e4 {
        return BFSType::Wrong;
    }
    let prod = u * y;
    if prod.iter().all(|&elt| (elt.abs() - 1.0).abs() < zthresh) {
        return BFSType::PM1;
    } else if prod.iter().all(|&elt| (elt.abs() - 1.0).abs() < zthresh
            || elt.abs() < zthresh) {
        return BFSType::PM10;
    }
    BFSType::Wrong
} //@}
// end BFS finder.@}

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

// Tableau{@
#[derive(Debug)] //{@
/// Error type for FlexTab.
//@}
enum FlexTabError { //{@
    SingularInput,
    GoodCols,
    LinIndep,
    NumZeroSlackvars,
    URowRedef,
    XVarsBothNonzero,
    StateStackExhausted,
    TooManyHops,
    Runout,
} //@}
impl error::Error for FlexTabError { //{@
    fn description(&self) -> &str {
        match self {
            &FlexTabError::SingularInput => "Input Y is not full rank",
            &FlexTabError::LinIndep => "Insuffient linearly independent columns",
            &FlexTabError::GoodCols => "Insufficient good columns",
            &FlexTabError::NumZeroSlackvars => "Num zero slackvars != 0",
            &FlexTabError::URowRedef => "U row redefinition",
            &FlexTabError::XVarsBothNonzero => "Xvars both nonzero",
            &FlexTabError::StateStackExhausted => "State stack exhausted",
            &FlexTabError::TooManyHops => "Too many hops",
            &FlexTabError::Runout => "Ran out of attempts", 
        }
    }
} //@}
impl fmt::Display for FlexTabError { //{@
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }
} //@}

#[derive(Clone)] //{@
/// Structure to maintain state during vertex hopping to support backtracking.
//@}
struct State { //{@
    x: Vec<f64>,
    u: na::DMatrix<f64>,
    uy: na::DMatrix<f64>,
    uybad: Option<na::DMatrix<f64>>,

    rows: na::DMatrix<f64>,
    vmap: Vec<usize>,

    grad: na::DMatrix<f64>,
    flip_grad: Vec<f64>,
    flip_grad_max: f64,
    flip_grad_min: f64,

    obj: f64,
    det_u: f64,
    vertex: Vertex,
} //@}
impl Default for State { //{@
    fn default() -> State {
        State {
            x: Vec::new(),
            u: na::DMatrix::from_column_slice(0, 0, &Vec::new()),
            uy: na::DMatrix::from_column_slice(0, 0, &Vec::new()),
            uybad: None,
            
            rows: na::DMatrix::from_column_slice(0, 0, &Vec::new()),
            vmap: Vec::new(),

            grad: na::DMatrix::from_column_slice(0, 0, &Vec::new()),
            flip_grad: Vec::new(),
            flip_grad_max: 0.0,
            flip_grad_min: 0.0,

            obj: 0.0,
            det_u: 0.0,
            vertex: Vertex::new(0),
        }
    }
} //@}
impl State { //{@
    fn new(u: na::DMatrix<f64>, uy: na::DMatrix<f64>, //{@
            uybad: Option<na::DMatrix<f64>>) -> State {
        let (n, k) = uy.shape();
        State {
            x: vec![0.0; 2 * (n*n + n*k)], 
            u: u,
            uy: uy,
            uybad: uybad,

            rows: na::DMatrix::from_column_slice(2 * n*k, 2 * (n*n + n*k) + 1,
                    &vec![0.0; (2 * n*k) * (2 * (n*n + n*k) + 1)]),
            vmap: vec![0; n*n],

            grad: na::DMatrix::from_column_slice(n, n, &vec![0.0; n*n]),
            flip_grad: vec![0.0; n*n],

            vertex: Vertex::new(n * n),

            ..Default::default()
        }
    } //@}

    fn copy_from(&mut self, other: &State) { //{@
        self.x.copy_from_slice(&other.x);
        self.vmap.copy_from_slice(&other.vmap);
        self.rows.copy_from(&other.rows);
        self.u.copy_from(&other.u);
        self.grad.copy_from(&other.grad);
        self.flip_grad.copy_from_slice(&other.flip_grad);
        self.uy.copy_from(&other.uy);
        match self.uybad {
            None => (),
            Some(ref mut uybad) => match other.uybad {
                Some(ref other_uybad) => uybad.copy_from(&other_uybad),
                None => panic!("Attempted copy from empty state uybad"),
            },
        }
        self.obj = other.obj;
        self.det_u = other.det_u;
        self.vertex.copy_from(&other.vertex);
    } //@}
} //@}

#[derive(Hash, Eq, Clone)] //{@
/// Structure for capturing vertices as a simple bool vector for use in
/// HashSets.
//@}
struct Vertex { //{@
    bits: Vec<bool>,
} //@}
impl Vertex { //{@
    fn new(nsq: usize) -> Vertex {
        Vertex { bits: vec![false; nsq] }
    }

    #[allow(dead_code)]
    fn from_vmap(vmap: &[usize]) -> Vertex {
        let mut vertex = Vertex::new(vmap.len());
        for (i, &v) in vmap.iter().enumerate() {
            vertex.bits[i] = v & 0x1 == 1;
        }
        vertex
    }
    
    fn copy_from(&mut self, other: &Vertex) {
        self.bits.copy_from_slice(&other.bits)
    }

    fn flip(&mut self, idx: usize) {
        self.bits[idx] = !self.bits[idx];
    }
} //@}
impl PartialEq for Vertex { //{@
    fn eq(&self, other: &Vertex) -> bool {
        if self.bits.len() != other.bits.len() {
            return false;
        }

        for i in 0 .. self.bits.len() {
            if self.bits[i] != other.bits[i] {
                return false;
            }
        }

        true
    }
} //@}

#[derive(Clone)] //{@
/// Structure for constraints that arise from extra "good" columns of Y beyond
/// the first n linearly independent columns.  These are referred to as type 3
/// constraints.
//@}
struct Constraint { //{@
    addends: Vec<(usize, f64)>,
    sum: f64,
}
impl Constraint { //{@
    fn new() -> Constraint {
        Constraint { addends: Vec::new(), sum: 0.0 }
    }

    fn push_addend(&mut self, xvar: usize, coeff: f64) {
        self.addends.push((xvar, coeff));
    }

    //{@
    /// Check if the constraint is still satisfied.
    //@}
    fn check(&self, x: &Vec<f64>, zthresh: f64) -> bool {
        if self.addends.len() == 0 { return true; }

        let total = self.addends.iter().fold(0.0, |total, &(xvar, coeff)| {
            total + coeff * x[xvar]
        });
        (total - self.sum).abs() < zthresh
    }
} //@}
impl Eq for Constraint {} //{@
impl PartialEq for Constraint {
    fn eq(&self, other: &Constraint) -> bool {
        if (self.sum - other.sum).abs() > ZTHRESH {
            return false;
        }

        if self.addends.len() != other.addends.len() {
            return false;
        }

        // Sort the addends by xvar.
        let mut s_addends = self.addends.clone();
        let mut o_addends = other.addends.clone();
        s_addends.sort_by_key(|&a| a.0);
        o_addends.sort_by_key(|&a| a.0);
        
        // Check the sorted addends.
        for i in 0 .. s_addends.len() {
            let ref s_add = s_addends[i];
            let ref o_add = o_addends[i];
            if s_add.0 != o_add.0 || (s_add.1 - o_add.1).abs() > ZTHRESH {
                return false;
           }
        }
        true
    }
} //@}
impl fmt::Display for Constraint { //{@
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} = {}",
                self.addends.iter().map(|&(xvar, coeff)|
                        format!("{}x_{}", coeff, xvar))
                    .collect::<Vec<_>>()
                    .join("  +  "),
                self.sum)
    }
} //@}
//@}

//{@
/// Principal tableau structure.
//@}
struct FlexTab { //{@
    verbose: u64,
    zthresh: f64,

    n: usize,
    k: usize,

    y: na::DMatrix<f64>,
    ybad: Option<na::DMatrix<f64>>,

    urows: Vec<Option<(usize, usize, usize)>>,
    extra_constr: Vec<Vec<Constraint>>,

    state: State,
    best_state: State,

    visited: HashSet<Vertex>,
    statestack: Vec<State>,
} //@}
impl Default for FlexTab { //{@
    fn default() -> FlexTab {
        FlexTab {
            verbose: 0,
            zthresh: 1e-9,

            n: 0,
            k: 0,

            y: na::DMatrix::from_column_slice(0, 0, &Vec::new()),
            ybad: None,

            urows: Vec::new(),
            extra_constr: Vec::new(),

            state: State { ..Default::default() },
            best_state: State { ..Default::default() },

            visited: HashSet::new(),
            statestack: Vec::new(),
        }
    }
} //@}
impl fmt::Display for FlexTab { //{@
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();

        // X-Values: 4 per line.
        for (i, &elt) in self.state.x.iter().enumerate() {
            s += &format!("x_{} = {:5.5}{}",
                    i, elt, if i % 4 == 3 { "\n" } else { " " });
        }
        s += &format!("\n");

        // Constraints: 1 per line.
        for i in 0 .. self.state.rows.nrows() {
            let r = self.state.rows.row(i);
            let mut line = Vec::new();
            for j in 0 .. r.len() - 1 {
                let v = r[j];
                if v.abs() > self.zthresh {
                    line.push(format!("{:5.2}x_{}", v, j));
                }
            }
            let line = line.join(" + ");
            s += &format!("{} == {:5.2}\n", line, r[r.len() - 1]);
        }

        // Type 3 constraints.
        println!("Type 3 Constraints:");
        for (rownum, ref cset) in self.extra_constr.iter().enumerate() {
            println!("\trow: {}", rownum);
            for ref c in cset.iter() {
                println!("\t\t{}", c);
            }
        }
        println!("\n");

        // urows
        s += "\nurows:\n";
        for (i, &u) in self.urows.iter().enumerate() {
            if let Some(u) = u {
                s += &format!("{:2} => {:?}", i, u);
            } else {
                s += &format!("{:2} => None", i);
            }
            s += if i % self.n == self.n - 1 { "\n" } else { " | " };
        }

        // vmap
        s += "\nvmap:\n";
        for (i, &v) in self.state.vmap.iter().enumerate() {
            s += &format!("{:2} => {:2}", i, v);
            s += if i % self.n == self.n - 1 { "\n" } else { " | " };
        }

        // current u, y, uy
        s += &format!("u = {:.5}y = {:.5}uy = {:.5}\n", self.state.u, self.y,
                self.state.uy);
        
        write!(f, "{}", s)
    }
} //@}
impl FlexTab { //{@
    fn new(u: &na::DMatrix<f64>, y: &na::DMatrix<f64>, zthresh: f64) //{@
            -> Result<FlexTab, FlexTabError> { 
        // Find all bad columns.
        let (bad_cols, uyfull) = FlexTab::find_bad_columns(&u, &y, zthresh);
        let n = y.nrows();
        let k = y.ncols() - bad_cols.len();

        // Ensure that there are at least n good columns; later we will check
        // for linear independence during tableau creation.
        if k < n {
            return Err(FlexTabError::GoodCols);
        }

        //let mut uy = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
        let mut uy: na::DMatrix<f64>;
        let mut y_used: na::DMatrix<f64>;
        let mut ybad = None;
        let mut uybad = None;

        if bad_cols.len() == 0 {
            // All columns are good, so we use them all.
            y_used = y.clone();
            uy = uyfull;

        } else {
            // Some columns are bad: only used for feasibility restrictions.
            y_used = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
            uy = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
            let mut _ybad = na::DMatrix::from_column_slice(n, bad_cols.len(),
                    &vec![0.0; n * bad_cols.len()]);
            let mut _uybad = na::DMatrix::from_column_slice(n, bad_cols.len(),
                    &vec![0.0; n * bad_cols.len()]);

            let mut cur_good = 0;
            let mut cur_bad = 0;
            for j in 0 .. y.ncols() {
                let src_col = y.column(j);
                if bad_cols.iter().any(|&bc| bc == j) {
                    _ybad.column_mut(cur_bad).copy_from(&src_col);
                    _uybad.column_mut(cur_bad).copy_from(&uyfull.column(j));
                    cur_bad += 1;
                } else {
                    y_used.column_mut(cur_good).copy_from(&src_col);
                    uy.column_mut(cur_good).copy_from(&uyfull.column(j));
                    cur_good += 1;
                }
            }
            assert!(cur_bad == bad_cols.len() && cur_good == k);

            ybad = Some(_ybad);
            uybad = Some(_uybad);
        }

        u.mul_to(&y_used, &mut uy);

        // Initialize mutable state.
        let state = State::new(u.clone(), uy, uybad);
        let best_state = state.clone();

        let mut ft = FlexTab {
            zthresh: zthresh,
            verbose: VERBOSE,
            
            n: n,
            k: k,

            y: y_used,
            ybad: ybad,

            urows: vec![None; n * n],
            extra_constr: vec![Vec::new(); n],

            state: state,
            best_state: best_state,

            ..Default::default()
        };
        ft.initialize_x();
        ft.set_constraints();
        Ok(ft)
    } //@}
    //{@
    /// Find the indices of all +bad+ columns, meaning there is at least one
    /// entry that is not \pm 1.
    /// Output: 1. Vector of indices of bad columns.
    ///         2. Product of UY for the full Y matrix.
    //@}
    fn find_bad_columns(u: &na::DMatrix<f64>, y: &na::DMatrix<f64>, //{@
            zthresh: f64) -> (Vec<usize>, na::DMatrix<f64>) {
        let (n, k) = y.shape();
        let mut bad_cols = Vec::new();
        let mut uyfull = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
        u.mul_to(&y, &mut uyfull);
        for j in 0 .. uyfull.ncols() {
            if uyfull.column(j).iter().any(|&e| (e.abs() - 1.0).abs() > zthresh) {
                bad_cols.push(j);
            }
        }
        (bad_cols, uyfull)
    } //@}
    //{@
    /// Number of columns such that uy = \pm 1.  This is NOT updated but
    /// reflects solely the original value.
    //@}
    #[allow(dead_code)]
    fn num_good_cols(&self) -> usize { //{@
        match self.ybad {
            Some(ref ybad) => self.y.ncols() - ybad.ncols(),
            None => self.y.ncols(),
        }
    } //@}
    #[allow(dead_code)]
    fn num_indep_cols(&self) -> usize { //{@
        // Determine the rank of Y.
        let svd = na::SVD::new(self.y.clone(), false, false);
        svd.singular_values.iter().filter(|&elt| *elt > self.zthresh).count()
    } //@}
    fn initialize_x(&mut self) { //{@
         for j in 0 .. self.state.u.ncols() {
             for i in 0 .. self.state.u.nrows() {
                 let idx = 2 * (self.n * i + j);
                 let elt = self.state.u.column(j)[i];
                 if elt >= 0.0 {
                     self.state.x[idx] = elt;
                 } else {
                     self.state.x[idx + 1] = -elt;
                 }

             }
         }
    } //@}
    //{@
    /// Set the constraints that will be used for vertex hoppinng.  These are
    /// only the constraints imposed by self.y, thus only the +good+ columns of
    /// Y.  The other constraints, imposed by the +bad+ columns of Y, are
    /// captured in self.extra_constr.
    //@}
    fn set_constraints(&mut self) { //{@
        let mut constraint = vec![(0, 0.0); self.n];
        let mut tab_row = 0;

        for i in 0 .. self.n {
            let x_var_base = i * self.n;
            for j in 0 .. self.k {
                {
                    let col = self.y.column(j);
                    // The i^th row uses variables in range: i*n .. i*(n+1).
                    // We are constraining |Uy|_\infty <= 1.
                    // Constraints are n-vector fo tuples [(varnum, coeff), ...  ].
                    for q in 0 .. self.n {
                        constraint[q] = (x_var_base + q, col[q]);
                    }
                }
                trace!("adding constraint: i = {}, j = {} c = {:?}", i, j, constraint);
                self.add_constraint_pair(tab_row, &constraint);
                tab_row += 2;
                trace!("ft =\n{}", self);
            }
        }
    } //@}
    //{@
    /// Add 2 new constraints based on the content of cset, which should be a
    /// slice of 2-tuples of the form (varnum: usize, coeff: f64).
    //@}
    fn add_constraint_pair(&mut self, tab_row: usize, cset: &[(usize, f64)]) { //{@
        // Setup the two new constraints based on the vars / coeffs given.
        for &(var, coeff) in cset.iter() {
            // Example: 3.1x_0 will become:
            //     tab_row    :  3.1x'_0 - 3.1x'_1 <= 1.
            //     tab_row +1 : -3.1x'_0 + 3.1x'_1 <= 1.
            self.state.rows.row_mut(tab_row)[2 * var] = coeff;
            self.state.rows.row_mut(tab_row)[2 * var + 1] = -coeff;
            self.state.rows.row_mut(tab_row + 1)[2 * var] = -coeff;
            self.state.rows.row_mut(tab_row + 1)[2 * var + 1] = coeff;
        }

        // Slack var coeffs both = 1.0.
        let zeroth_slack = 2 * self.n * self.n + tab_row;
        let first_slack = zeroth_slack + 1;
        self.state.rows.row_mut(tab_row)[zeroth_slack] = 1.0;
        self.state.rows.row_mut(tab_row + 1)[first_slack] = 1.0;

        // RHS of both constraints = 1.0.
        let rhs = self.state.rows.ncols() - 1;
        self.state.rows.row_mut(tab_row)[rhs] = 1.0;
        self.state.rows.row_mut(tab_row + 1)[rhs] = 1.0;

        // Need to determine values of the slack variables for this pair of
        // constraints.  Compute the value of the LHS of the constraint.  One of
        // the pair will be tight, so slack will be 0; other slack will be 2.
        // Compute: val = \sum_{2n LHS vars in this constr_set} a_{ij} * x_j.
        let x_lowbound = (2 * self.n) * (tab_row / (2 * self.k));
        let x_highbound = x_lowbound + 2 * self.n;
        let val = (x_lowbound .. x_highbound).fold(0.0, |val, j|
                val + self.state.x[j] * self.state.rows.row(tab_row)[j]);
        // Set slack var values so LHS = 1.0.
        self.state.x[zeroth_slack] = 1.0 - val;
        self.state.x[first_slack] = 1.0 + val;
    } //@}

    //{@
    /// Create simplex-style tableau, which will be stored in self.rows.  Note
    /// that self.X is of length 2 * self.nk.  There are 2 * self.nk overall
    /// constraints in |UY|_\infty \leq 1.
    //@}
    fn to_simplex_form(&mut self) -> Result<(), FlexTabError> { //{@
        // Zero out any entries in self.x that are solely float imprecision.
        let num_x_from_u = 2 * self.n * self.n;

        trace!("ft = {}", self);
        // There are n groups, each of k equation pairs, and within a group
        // every equation pair uses the same set of x variables.
        for groupnum in 0 .. self.n {
            // Grab each [of the k] equation pairs individually.  Idea is to
            // iterate over all still-available basic vars, then if one exists
            // such that self.rows[row, var] > zthresh, then use var as the
            // basic var for this eq_set [where row = 2 * eq_set].
            let mut avail_basic = (groupnum * self.n .. (groupnum + 1) * self.n)
                .collect::<HashSet<_>>();

            for j in 0 .. self.k {
                // eq_pair = pair of equations we are currently focused on.
                let eq_pair = groupnum * self.k + j;

                // Exactly one of eq_pair should have a zero slack variable.
                let lowzero = self.state.x[num_x_from_u + 2 * eq_pair].abs()
                    < self.zthresh;
                let highzero = self.state.x[num_x_from_u + 2 * eq_pair + 1].abs()
                    < self.zthresh;
                if !(lowzero ^ highzero) {
                    println!("num zero slackvars in pair {} != 1", eq_pair);
                    return Err(FlexTabError::NumZeroSlackvars);
                }
                let zeroslack = if lowzero { 2 * eq_pair } else { 2 * eq_pair + 1 };

                // Find available basic vars with nonzero coeff in this pair.
                let uvar = {
                    let useable = avail_basic.iter()
                        .filter(|&v| self.state.rows.row(2 * eq_pair)[2 * v].abs()
                                > self.zthresh)
                        .collect::<Vec<_>>();
                    match useable.first() {
                        Some(uvar) => Some(**uvar),
                        None => None,
                    }
                };
                        
                // If we have an available uvar then set a basic variable;
                // otherwise, simply add one row to the other.
                if let Some(uvar) = uvar {
                    // Use the row where slack = 0 to turn an original U-var
                    // into a basic var.  For ease use the original x_j, so we
                    // make basic var from whichever of x_{2j}, x_{2j+1} != 0.
                    avail_basic.remove(&uvar);
                    let x_basic = zeroslack;
                    self.set_basic(uvar, x_basic)?;
                } else {
                    let tgtrow = zeroslack;
                    let srcrow = tgtrow ^ 0x1;
                    self.add_row_multiple(tgtrow, srcrow, 1.0);
                }
                trace!("ft = {}", self);
            }
        }
        self.tableau_mappings()?;
        Ok(())
    } //@}
    //{@
    /// Create a simplex-style basic variable from the original x_j.  Note that
    /// the original x_j \in R is now divided into the difference of variables
    /// \in R^+: x_{2j} and x_{2j+1}.  We make a basic variable from whichever
    /// of x_{2j}, x_{2j+1} is nonzero.  The selected variable has its
    /// coefficient in pivot_row set to 1, and zeroed everywhere else.
    /// INPUT:  j = original x-var: the new basic var will be whichever of
    ///             x_{2j}, x_{2j+1} is nonzero.
    ///         pivot_row = row in which the selected new basic var should have
    ///             coefficient set to 1.
    /// OUTPUT: true on success.
    //@}
    fn set_basic(&mut self, j: usize, pivot_row: usize) //{@
            -> Result<(), FlexTabError> {
        // basic var = whichever of x_{2j}, x_{2j+1} != 0.
        let lownz = self.state.x[2*j].abs() > self.zthresh;
        let highnz = self.state.x[2*j + 1].abs() > self.zthresh;
        if lownz && highnz {
            error!("xvars {}, {} both != 0", 2*j, 2*j + 1);
            return Err(FlexTabError::XVarsBothNonzero);
        }
        let tgtvar = if lownz { 2 * j } else { 2 * j + 1 };

        // Make coeff of pivot_row[tgtvar] = 1.
        let divisor = self.state.rows.row(pivot_row)[tgtvar];
        self.div_row_float(pivot_row, divisor);

        // Eliminate tgtvar from every other row.
        let baserow = 2 * (j / self.n * self.k);
        let limrow = baserow + 2 * self.k;
        for i in baserow .. limrow {
            if i == pivot_row { continue; }
            let mult = -1.0 * self.state.rows.row(i)[tgtvar];
            self.add_row_multiple(i, pivot_row, mult);
        }

        Ok(())
    } //@}
    fn tableau_mappings(&mut self) -> Result<(), FlexTabError> { //{@
        // Each constraint set corresponds to one row of U.
        for cset in 0 .. self.n {
            let mut vvars:HashSet<usize> = HashSet::new();
            for row in 2 * self.k * cset .. 2 * self.k * (cset + 1) {
                let (uvars, slackvars) = self.categorize_row(row);
                trace!("row = {}, uvars = {:?}, slackvars = {:?}",
                        row, uvars, slackvars);

                // If there are any uvars, there must be 2, both from same Xvar.
                if uvars.len() > 0 {
                    if uvars.len() != 2 || uvars[0] >> 1 != uvars[1] >> 1 {
                        // This fails if not enough lin indep cols.
                        warn!("uvars: row = {}, uvars = {:?}", row, uvars);
                        return Err(FlexTabError::LinIndep);
                    }
                    let uvar = uvars[0] >> 1;
                    match self.urows[uvar] {
                        Some(_) => {
                            error!("urows[{}] redef in row {}", uvar, row);
                            return Err(FlexTabError::URowRedef);
                        },
                        None => {
                            // Tuple is (row, xplus, xminus), where +row+ is the
                            // actual tableau row, +xplus+ is the tableau column
                            // in which the variable +uvar+ has a + coeff.
                            self.urows[uvar] = 
                                if self.state.rows.row(row)[2 * uvar] > 0.0 {
                                    Some((row, 2 * uvar, 2 * uvar + 1))
                                } else {
                                    Some((row, 2 * uvar + 1, 2 * uvar))
                                };
                        }
                    };
                    // All of the RHS vars in 2-uvar eqns are "normal" slacks.
                    for v in slackvars { vvars.insert(v); }

                } else {
                    // If exactly 2 slackvars and both correspond to same base
                    // eqn, then this is dealt with in vvars.  Otherwise, this
                    // gives us a type 3 constraint.
                    let x_u_count = 2 * self.n * self.n;
                    let slack_eqn = |q: usize| (q - x_u_count) >> 1;
                    if slackvars.len() != 2
                          || slack_eqn(slackvars[0]) != slack_eqn(slackvars[1]) {
                        self.add_extra_constr(cset, row, &slackvars);
                    }
                }
            }

            let mut vvars = vvars.iter().map(|&v| v).collect::<Vec<usize>>();
            vvars.sort();
            for (i, &v) in vvars.iter().enumerate() {
                self.state.vmap[cset * self.n + i] = v;
            }
        }
        Ok(())
    } //@}
    //{@
    /// Return arrays of variables that are:
    ///  1. uvars: X vars derived from U vars (potential basic)
    ///  2. slackvars: X vars that are slackvars
    //@}
    fn categorize_row(&self, rownum: usize) -> (Vec<usize>, Vec<usize>) { //{@
        // Determine which series we are in, where the series is the row of U
        // from which these constraints are derived.
        let g = rownum / (2 * self.k);
        let row = self.state.rows.row(rownum);

        // The 2n U-vars are those from the relevant row.
        let low_u = 2 * self.n * g;
        let high_u = low_u + 2 * self.n;
        let uvars = (low_u .. high_u)
            .filter(|&j| row[j].abs() > self.zthresh)
            .collect::<Vec<_>>();

        // The 2k potential slack vars are those involving these same U rows.
        let low_s = (2 * self.n * self.n) + (2 * self.k * g);
        let high_s = low_s + 2 * self.k;
        let slackvars = (low_s .. high_s)
            .filter(|&j| row[j].abs() > self.zthresh)
            .collect::<Vec<_>>();

        (uvars, slackvars)
    } //@}
    fn add_extra_constr(&mut self, csetnum: usize, rownum: usize, //{@
            slackvars: &Vec<usize>) {
        let lastcol = self.state.rows.ncols() - 1;
        let row = self.state.rows.row(rownum);
        let sum = row[lastcol];

        // Create constraint, set sum.
        let mut con = Constraint::new();
        con.sum = sum;

        // Push all addends to this constraint.
        for &v in slackvars {
            con.push_addend(v, row[v]);
        }

        // Add fully formed constraint to relevant extra constraint set.
        self.extra_constr[csetnum].push(con);
    } //@}

    fn snapshot(&mut self) { //{@
        self.statestack.push(self.state.clone());
    } //@}
    fn restore(&mut self, pop: bool) -> bool { //{@
        if pop {
            if let Some(state) = self.statestack.pop() {
                self.state.copy_from(&state);
                true
            } else {
                false
            }
        } else {
            if let Some(state) = self.statestack.last() {
                self.state.copy_from(&state);
                true
            } else {
                false
            }
        }
    } //@}

    fn solve(&mut self) -> Result<(), FlexTabError> { //{@
        self.to_simplex_form()?;
        trace!("Initial simplex form:\n{}\n", self);
        self.hop()?;
        Ok(())
    } //@}
    fn hop(&mut self) -> Result<(), FlexTabError> { //{@
        self.mark_visited();
        debug!("Initial obj = {:.5}, init u =\n{:.3}", self.state.obj, self.state.u);

        // Principal vertex hopping loop.
        loop {
            if self.verbose & VERBOSE_HOP != 0 {
                println!("HOP loop top: visited = {}", self.visited.len());
            }

            // Check if this vertex is valid: if not, backtrack.
            if !self.eval_vertex() {
                debug!("Bad vertex, attempting backtrack");
                if !self.restore(true) {
                    return Err(FlexTabError::StateStackExhausted);
                }
                continue;
            }
            // Check if this is the best vertex we have seen.
            if self.state.obj > self.best_state.obj {
                self.best_state.copy_from(&self.state);
            }
            // Check if this vertex is a global optimum.
            if self.is_done() {
                debug!("End obj = {:.5}", self.state.obj);
                break;
            }

            debug!("obj = {:.5}\nflip_grad =\n{}uy = {:.5}u = {:.5}\n",
                    self.state.obj, self.print_flip_grad(), self.state.uy,
                    self.state.u);
            if self.visited.len() >= 2 * self.n * self.n {
                return Err(FlexTabError::TooManyHops);
            }
            let (flip_idx, effect) = self.search_neighbors();
            match flip_idx {
                None => {
                    info!("Nowhere at all to go, attempting backtrack");
                    if !self.restore(true) {
                        return Err(FlexTabError::StateStackExhausted);
                    }
                },
                Some(idx) => {
                    // Take snapshot, flip idx, mark new vertex visited.
                    info!("flipping vertex: {}", idx);
                    self.snapshot();
                    if self.verbose & VERBOSE_HOP != 0 {
                        println!("Hop {}", if effect > self.zthresh
                                { "++" } else { "==" });
                    }
                    self.flip(idx);
                    self.mark_visited();
                }
            };
        }
        Ok(())
    } //@}
    fn search_neighbors(&mut self) -> (Option<usize>, f64) { //{@
        let mut best_idx = None;
        let mut best_effect = std::f64::MIN;

        for i in 0 .. self.n {
            for j in 0 .. self.n {
                let v = self.n * i + j;
                let effect = self.state.flip_grad[v];
                if !self.is_flip_visited(v)
                        && !effect.is_infinite()
                        && effect > best_effect {
                    best_idx = Some(v);
                    best_effect = effect;
                }
            }
        }

        (best_idx, best_effect)
    } //@}

    //{@
    /// Flip the variable U_v between {-1, +1}.  Since the variable U_v is
    /// represented as the difference X_{2v} - X_{2v+1}, this amounts to
    /// changing which of these variables is basic, so the variable that
    /// currently has value 0 will be the entering variable, and the one with
    /// value 2 will be the leaving variable.
    /// We are flipping the value of UY at flat index v, where flat indices are
    /// in row-major order.
    //@}
    fn flip(&mut self, v: usize) { //{@
        // Set vertex used for visited hash.
        self.state.vertex.flip(v);
        
        // The self.vmap holds the index v such that x_v is the currently 0
        // slack variable that needs to be flipped to 2 in order to exec pivot.
        let old_zero = self.state.vmap[v];
        let new_zero = old_zero ^ 0x1;
        self.state.vmap[v] = new_zero;
        self.state.x[old_zero] = 2.0;
        self.state.x[new_zero] = 0.0;

        // To exec the pivot, we must pivot each of the (n) u-values.
        let base = (v / self.n) * self.n;
        let top = base + self.n;
        trace!("IN +flip+ ft =\n{}", self);
        for i in base .. top {
            let (maniprow, xplus, xminus) = self.urows[i].unwrap();
            if self.verbose & VERBOSE_FLIP != 0 {
                println!("i = {}, maniprow = {}", i, maniprow);
            }

            // Each maniprow will have the old_zero variable: get coefficient.
            let old_zero_coeff = self.state.rows.row(maniprow)[old_zero];

            // Execute pivot.
            let mut row = self.state.rows.row_mut(maniprow);
            let rhs = row.len() - 1;
            row[rhs] -= old_zero_coeff * 2.0;
            row[old_zero] = 0.0;
            row[new_zero] = -old_zero_coeff;

            // Set the relevant value of x.
            if row[rhs] >= 0.0 {
                self.state.x[xplus] = row[rhs];
                self.state.x[xminus] = 0.0;
            } else {
                self.state.x[xplus] = 0.0;
                self.state.x[xminus] = -row[rhs];
            }
        }
    } //@}
    fn is_flip_visited(&mut self, idx: usize) -> bool { //{@
        self.state.vertex.flip(idx);
        let ans = self.visited.contains(&self.state.vertex);
        self.state.vertex.flip(idx);
        ans
    } //@}
    fn eval_vertex(&self) -> bool { //{@
        if self.state.obj < -10.0 || self.state.flip_grad_max > 50.0 {
            return false;
        }

        // Type 3 constraints.
        for (row, ref cset) in self.extra_constr.iter().enumerate() {
            for (cnum, ref c) in cset.iter().enumerate() {
                // Constraint contains set of relevant columns.  We need to pass
                // in the row.
                if !c.check(&self.state.x, self.zthresh) {
                    info!("type 3 check fail: row = {}, cnum = {}", row, cnum);
                    return false;
                }
            }
        }

        // Full UY feasibility: check _bad_cols, good enforced by tableau.
        if let Some(ref uybad) = self.state.uybad {
            if uybad.iter().any(|&e| e.abs() > 1.0 + self.zthresh) {
                return false;
            }
        }

        true
    } //@}
    fn mark_visited(&mut self) { //{@
        self.set_u();
        self.set_obj();
        self.set_uy();
        self.set_grad();
        self.set_flip_grad();
        self.set_uybad();
        self.visited.insert(self.state.vertex.clone()); // Must be set in +flip+
    } //@}

    fn add_row_multiple(&mut self, tgtrow: usize, srcrow: usize, mult: f64) { //{@
        let mut src = self.state.rows.row(srcrow).iter()
            .map(|e| *e)
            .collect::<Vec<f64>>();
        if mult != 1.0 {
            for e in src.iter_mut() { *e *= mult; }
        }

        let ref mut tgt = self.state.rows.row_mut(tgtrow);
        for j in 0 .. src.len() {
            tgt[j] += src[j];
        }
    } //@}
    fn div_row_float(&mut self, tgtrow: usize, divisor: f64) { //{@
        for e in self.state.rows.row_mut(tgtrow).iter_mut() {
            *e /= divisor;
        }
    } //@}
    //{@
    /// Reconstruct U from first 2 * self.n^2 entries of X.
    //@}
    fn set_u(&mut self) { //{@
        for j in 0 .. self.state.u.ncols() {
            for i in 0 .. self.state.u.nrows() {
                let idx = 2 * (self.n * i + j);
                let xplus = self.state.x[idx];
                let xminus = self.state.x[idx + 1];
                self.state.u.column_mut(j)[i] = xplus - xminus;
            }
        }

        self.state.det_u = self.state.u.determinant().abs();
    } //@}
    fn set_obj(&mut self) { //{@
        if self.state.det_u < self.zthresh {
            self.state.obj = std::f64::NEG_INFINITY;
        } else {
            self.state.obj = self.state.det_u.ln();
        }
    } //@}
    fn set_grad(&mut self) { //{@
        let u = self.state.u.clone();
        if let Some(inv) = u.try_inverse() {
            let inv = inv.transpose();
            self.state.grad.copy_from(&inv);
        } else {
            // Some sort of error here / maybe a Result?.
            self.state.grad.iter_mut().for_each(|e| *e = 0.0);
        }
    } //@}
    fn set_uy(&mut self) { //{@
        self.state.u.mul_to(&self.y, &mut self.state.uy);
    } //@}
    fn set_uybad(&mut self) { //{@
        if let Some(ref mut uybad) = self.state.uybad {
            if let Some(ref ybad) = self.ybad {
                self.state.u.mul_to(&ybad, uybad);
            }
        }
    } //@}
    fn set_flip_grad(&mut self) { //{@
        self.state.flip_grad_max = std::f64::MIN;
        self.state.flip_grad_min = std::f64::MAX;

        // Need to set flip_gradient for all self.n^2 elts of self_grad.
        for i in 0 .. self.n {
            for j in 0 .. self.n {
                let v = self.n * i + j; // index being evaluated
                let mut total_effect = 0.0;

                // Iterate over every U-var in the same row as the vertex v that
                // we are evalutating.
                for u_j in 0 .. self.n {
                    let u = self.n * i + u_j;
                    // self.urows[u] contais which +base+ row has this U-var.
                    // Here we need the coefficient sign, not X-var value sign.
                    let urow = self.urows[u].expect("ERROR: urow not set").0;
                    let sign = self.state.rows.row(urow)[2 * u];

                    // Get coefficient of the slackvar that will go 0 -> 2.
                    let coeff = self.state.rows.row(urow)[self.state.vmap[v]];

                    // Calc effect of moving this slackvar on the U-value, then
                    // multiply this delta by the gradient.
                    let delta_u = -2.0 * coeff * sign;
                    let obj_effect = delta_u * self.state.grad.row(i)[u_j];
                    total_effect += obj_effect;
                }

                // Set values.
                self.state.flip_grad[v] = total_effect;
                if total_effect < self.state.flip_grad_min {
                    self.state.flip_grad_min = total_effect;
                }
                if total_effect > self.state.flip_grad_max {
                    self.state.flip_grad_max = total_effect;
                }
            }
        }
    } //@}
    #[allow(dead_code)]
    fn print_flip_grad(&self) -> String { //{@
        let mut s = String::new();
        for (i, &e) in self.state.flip_grad.iter().enumerate() {
            s += &format!("{:-3}: {:.3}\n", i, e);
        }
        s
    } //@}

    fn is_done(&self) -> bool { //{@
        match self.state.uybad {
            Some(ref uybad) => uybad.iter().all(|&elt|
                    (elt.abs() - 1.0).abs() < self.zthresh),

            None => match self.n {
                2 => self.is_done_2(),
                3 => self.is_done_3(),
                4 => self.is_done_4(),
                5 => self.is_done_5(),
                6 => self.is_done_6(),
                7 => self.is_done_7(),
                8 => self.is_done_8(),
                _ => false,
            }
        }
    } //@}
    fn is_done_2(&self) -> bool { //{@
        self.state.det_u > self.zthresh
    } //@}
    fn is_done_3(&self) -> bool { //{@
        self.state.det_u > self.zthresh
    } //@}
    //{@
    /// All neighbors have det 8.
    //@}
    fn is_done_4(&self) -> bool { //{@
        (self.state.flip_grad_max + 0.50).abs() < 1e-2
            && (self.state.flip_grad_min + 0.50).abs() < 1e-2
    } //@}
    //{@
    /// 16, 16, 16, 16, 16,
    /// 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    /// 32, 32, 32, 32, 32, 32, 32, 32, 32, 32
    //@}
    fn is_done_5(&self) -> bool { //{@
        // Done iff a fraction of 3.
        let g = self.state.flip_grad[0] * 3.0;
        (g - g.round()).abs() < 1e-5
    } //@}
    //{@
    /// 64, 64, 64, 64, 64, 64,
    /// 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96,
    /// 128, 128, 128, 128, 128, 128, 128, 128, 128,
    /// 128, 128, 128, 128, 128, 128, 128, 128, 128
    //@}
    fn is_done_6(&self) -> bool { //{@
        
        // Done iff a fraction of 5.
        let g = self.state.flip_grad[0] * 5.0;
        (g - g.round()).abs() < 1e-5
    } //@}
    fn is_done_7(&self) -> bool { //{@
        self.state.flip_grad_max < -self.zthresh
    } //@}
    //{@
    /// All neighbors have det 3072.
    //@}
    fn is_done_8(&self) -> bool { //{@
        (self.state.flip_grad_max + 0.25).abs() < 1e-2
            && (self.state.flip_grad_min + 0.25).abs() < 1e-2
    } //@}
} //@}
// end Tableau@}
// end principal functions@}

// Misc functions{@
// structures //{@
#[derive(Eq)]
//{@
/// Data structure for a vector of determinants.
//@}
struct Detvec {
    dets: Vec<u32>,
}

impl PartialEq for Detvec { //{@
    //{@
    /// Must both be sorted prior to calling any function that uses +eq+.
    //@}
    fn eq(&self, other: &Detvec) -> bool {
        if self.dets.len() != other.dets.len() {
            return false;
        }

        for i in 0 .. self.dets.len() {
            if self.dets[i] != other.dets[i] {
                return false;
            }
        }

        true
    }
} //@}

impl Ord for Detvec { //{@
    fn cmp(&self, other: &Detvec) -> Ordering {
        if self.dets.len() < other.dets.len() {
            return Ordering::Less;
        } else if self.dets.len() > other.dets.len() {
            return Ordering::Greater;
        }

        for i in 0 .. self.dets.len() {
            if self.dets[i] < other.dets[i] {
                return Ordering::Less;
            } else if self.dets[i] > other.dets[i] {
                return Ordering::Greater;
            }
        }

        Ordering::Equal
    }
} //@}

impl PartialOrd for Detvec { //{@
    fn partial_cmp(&self, other: &Detvec) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
} //@}

impl Detvec { //{@
    fn new(v: &Vec<u32>) -> Detvec {
        let mut dv = Detvec { dets: v.clone() };
        dv.dets.sort();
        dv
    }
} //@}

impl fmt::Display for Detvec { //{@
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = &format!("[{}]", 
                self.dets.iter().map(|e| format!("{}", e))
                .collect::<Vec<_>>()
                .join(", "));
        write!(f, "{}", s)
    }
} //@}
// end structures //@}
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

#[allow(dead_code)] //{@
/// Read a set of matrices from file +fname+, return as a vector.
//@}
fn read_mtxset_file(fname: &str) -> Vec<na::DMatrix<f64>> { //{@
    let mut ret = Vec::new();
    let f = File::open(fname).unwrap();
    let reader = BufReader::new(f);
    let blank = Regex::new(r"^\s*$").unwrap();
    let numre = Regex::new(r"(\-?\d+)").unwrap(); 

    let data2mtx = |data: &Vec<f64>, numlines: usize| {
        if data.len() == 0 {
            return None;
        }
        if data.len() % numlines != 0 {
            println!("Input error: {} / {}", data.len(), numlines);
            return None;
        }
        Some(na::DMatrix::from_row_slice(numlines, data.len() / numlines, &data))
    };

    let mut data = Vec::new();
    let mut numlines = 0;
    for line in reader.lines().map(|l| l.unwrap()) {
        if blank.is_match(&line) {
            if let Some(mtx) = data2mtx(&data, numlines) {
                ret.push(mtx);
            }
            data = Vec::new();
            numlines = 0;
            continue;
        }

        let mut values = numre.captures_iter(&line)
            .map(|c| c[0].parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        data.append(&mut values);
        numlines += 1;
    }
    if let Some(mtx) = data2mtx(&data, numlines) {
        ret.push(mtx);
    }

    ret
} //@}
#[allow(dead_code)] //{@
/// Read a matrix from file +fname+ and return it.
//@}
fn read_mtx_file(fname: &str) -> na::DMatrix<f64> { //{@
    let f = File::open(fname).unwrap();
    let reader = BufReader::new(f);
    let re = Regex::new(r"(\-?\d+\.\d+)").unwrap();
    let mut data = Vec::new();
    let mut nrows = 0;
    let mut ncols: Option<usize> = None;

    for line in reader.lines().map(|l| l.unwrap()) {
        let mut values = re.captures_iter(&line)
            .map(|c| (&c[0]).parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        if values.len() == 0 { continue; }
        match ncols {
            None => ncols = Some(values.len()),
            Some(n) => assert_eq!(values.len(), n),
        };
        data.append(&mut values);
        nrows += 1;
    }

    na::DMatrix::from_row_slice(nrows, ncols.unwrap(), &data)
} //@}

#[allow(dead_code)] //{@
/// For each matrix in the file +fname+, calculate the determinant pattern of
/// every neighboring matrix.  Print each unique determinant pattern.
//@}
fn neighbor_det_pattern(fname: &str) { //{@
    // Simple function for calculating determinant.
    let calcdet: fn(m: &na::DMatrix<f64>) -> u32 = 
        |m| m.determinant().abs().round() as u32;

    // Set number of threads to use.
    let nthreads = 4;
    // Read in all of the matrices from file.
    let maxmats = read_mtxset_file(fname);
    // Create shared data structure.
    let detvecs = Arc::new(Mutex::new(BTreeSet::new()));

    let timer = std::time::Instant::now();
    let mtx_per_thread = maxmats.len() / nthreads;
    let mut threads = Vec::with_capacity(nthreads);

    // Spawn threads.
    for tnum in 0 .. nthreads {
        let detvecs = detvecs.clone(); // clones the Arc
        let maxmats = maxmats.clone();
        let child = thread::spawn(move || {
            for i in (tnum * mtx_per_thread) .. ((tnum + 1) * mtx_per_thread) {
                println!("Thread: {}, mtx: {}", tnum, i);
                let mtx = &maxmats[i];
                let neighbor_dets = see_neighbors(&mtx, calcdet);
                let mut detvecs = detvecs.lock().unwrap();
                (*detvecs).insert(Detvec::new(&neighbor_dets));
            }
        });
        threads.push(child);
    }

    for thr in threads { thr.join().unwrap(); }
    let nanos = timer.elapsed().subsec_nanos();

    let detvecs = detvecs.lock().unwrap();
    println!("Detvecs:");
    for dv in detvecs.iter() { println!("{}", dv); }
    println!("time = {:.6}", nanos as f64 * 1e-9);
} //@}
#[allow(dead_code)] //{@
/// For each matrix in {-1, +1} ^ {n x n} for specified parameter _n_, calculate
/// the determinant of the matrix and the determinant pattern of all neighboring
/// matrices.  For each matrix determinant, print each unique neighbor pattern.
/// This is done very inefficiently: do not call with n > 5.
//@}
fn neighbor_det_pattern_all(n: usize) { //{@
    if n > 5 { panic!("Cannot call function with n > 5"); }

    // Foreach possible determinant value, need the set of possible detvecs.
    let mut dets = HashMap::new();

    let calcdet: fn(m: &na::DMatrix<f64>) -> u32 =
        |m| m.determinant().abs().round() as u32;

    // Generator for each \pm1 matrix of size n.
    let mut gen = gen_all_pm1(n);
    let mut num_mtx = 0;
    while let Some(mtx) = gen() {
        num_mtx += 1;
        if num_mtx & 0xf_ffff == 0 { println!("num = {:x}", num_mtx); }
        let det = calcdet(&mtx); // Get the determinant of this matrix.
        let neighbor_dets = see_neighbors(&mtx, calcdet);
        if !dets.contains_key(&det) {
            dets.insert(det, BTreeSet::new());
        }
        dets.get_mut(&det).unwrap().insert(Detvec::new(&neighbor_dets));
    }

    let mut keys = dets.keys().collect::<Vec<_>>();
    keys.sort();
    for &det in keys.iter() {
        println!("det = {}\n{}\n",
                det,
                dets.get(det).unwrap().iter()
                    .map(|ref dv| format!("{}", dv))
                    .collect::<Vec<_>>()
                    .join("\n")
                );
    }
} //@}
//{@
/// Iterate over each neighbor of +mtx+.  For each neighbor, call the +process+
/// function on that neighbor and store the result.
/// Return a vector containing the results of calling +process+ on each
/// neighbor.
//@}
fn see_neighbors<T>(mtx: &na::DMatrix<f64>, process: fn(&na::DMatrix<f64>) -> T) //{@
        -> Vec<T>
        where T: Ord + PartialOrd {
    let mut mtx = mtx.clone();
    let mut dets = Vec::new();
    let (nrows, ncols) = mtx.shape();

    // Iterate over each neighbor.
    for i in 0 .. nrows {
        for j in 0 .. ncols {
            // Flip to neighbor.
            let orig = mtx.row(i)[j];
            mtx.row_mut(i)[j] = if orig == -1.0 { 1.0 } else { -1.0 };

            // Call the specified _process_ function, push the result.
            dets.push(process(&mtx));

            // Flip back.
            mtx.row_mut(i)[j] = orig;
        }
    }
    dets
} //@}
// end Misc functions@}
