/*
 * This code is proprietary information and property of Stanford University.
 * Do not share without permission of the authors.  This code may only be
 * used for educational purposes.  
 *
 * Authors: Jonathan Perlstein (jrperl@cs.stanford.edu)
 *          Thomas Dean (trdean@stanford.edu)
 *
 */

// imports{@
extern crate nalgebra as na;
extern crate rand;
extern crate regex;
#[macro_use]
extern crate log;
extern crate env_logger;

use std::io::stdout;
use std::io::Write;

mod matrix;
mod testlib;
mod tableau;
mod dynamic;

use testlib::TrialResults;
use tableau::FlexTabError;
use tableau::FlexTab;
use dynamic::BFSType;

// end imports@}
// constants{@
const ZTHRESH: f64 = 1e-9;
// end constants@}

#[allow(dead_code)]
/// This just tests the BFS finding with a fixed input.  The BFS solver will return
/// something infeasible half the time
fn bfs_test() {
    let y = na::DMatrix::from_row_slice( 4,6,
            &vec![
            -0.85105277, -0.63492999, 3.41237287, 1.20811145, 0.35892148, 2.41836825,
            3.96976924, -0.73519243, -1.04010774, 0.12971196, 1.03251896, -2.80775547,
            3.12355480, 2.95773796, 0.67356399, 0.87056107, 2.94202708, 0.68919778,
            -2.31709338, 0.66587824, 1.21174544, 2.38172050, -1.41046808, 3.28820386
            ]);

    let ui = na::DMatrix::from_row_slice( 4,4,
           &vec![
            -0.06779237, -0.08776801,  0.03347982, -0.04695821,
            -0.01830087, -0.05418955, -0.09233134,  0.06187484,
            0.06205623, -0.02276697, -0.05832655, -0.08862112,
            0.08272422, -0.06683513,  0.05076455,  0.04168607
           ]);

    println!("UY={}", ui.clone()*y.clone());
    let bfs = dynamic::find_bfs(&ui, &y).unwrap();
    println!("UY={:.3}", bfs * y );
}

// Principal functions{@
// Runnable functions{@

#[allow(dead_code)]//{@
/// Run the BFS finder multiple times over multiple different base matrices.
//@}
fn many_bfs(reps: usize) { //{@
    /*
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
    */

    let dims: Vec<(usize,usize)> = (6..20).filter(|x| x % 2 == 0).map(|x| (4,x) ).collect();

    //let mut rng = rand::thread_rng();
    //let mut results = HashMap::new();
    for i in 0 .. dims.len() {
        let mut good = 0;
        let mut zero = 0;
        let mut other = 0;

        for _ in 0 .. reps {
            //let basemtx = rng.choose(&xmats).unwrap();
            let x = matrix::get_matrix(&dims[i .. i+1]);
            //let x = basemtx.fill();
            let (_a, y) = trial(&x, false);
            let u_i = matrix::rand_init(&y);
            //println!("a = {}\ny = {}\nUi = {}", a, y, u_i);
            let bfs = dynamic::find_bfs(&u_i, &y).unwrap();
            //let uy = bfs.clone() * y.clone();
            //println!("UY = {}\n", uy);

            let (g,z,o) = count_bfs_entry(&bfs, &y, 1e-3);
            good += g; zero += z; other += o;

            //if o > 0 {
            //    println!( "UY = {:.13}", bfs.clone() * y.clone() );
            //    println!( "Y = {:.13}", y.clone() );
            //    println!( "U_i = {:.13}", u_i.clone() );
            //}

            //let res = verify_bfs(&bfs, &y, ZTHRESH);
            //if res == BFSType::Wrong {
            //    let uy = bfs.clone() * y.clone();
            //    println!("BFS Fail: a = {:.6}y = {:.6}u_i = {:.6}u = {:.6}\nuy = {:.6}",
            //            a, y, u_i, bfs, uy);
            //}
            //let count = results.entry(res).or_insert(0);
            //*count += 1;
        }
        let total = reps * dims[i].0 * dims[i].1;

        print!("({},{}):\t", dims[i].0, dims[i].1);
        println!("Good = {:.4}, Zero = {:.3e}, Other = {:.3e}", (good as f64) / (total as f64), 
                 (zero as f64) / (total as f64), 
                 (other as f64) / (total as f64) );
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
/// Old centering code
fn near_pm_one( x: f64, tol: f64 ) -> f64 {
    if ( x - 1f64 ).abs() < tol {
        return x - 1f64;
    } else if ( x + 1f64 ).abs() < tol {
        return x + 1f64;
    }

    0f64
}

#[allow(dead_code)]
/// If x is within tol of {-1,0,1}, then return how far off it is
/// otherwise return 0
fn near_pm_one_or_zero( x: f64, tol: f64 ) -> f64 {
    if ( x - 1f64 ).abs() < tol {
        return x - 1f64;
    } else if ( x + 1f64 ).abs() < tol {
        return x + 1f64;
    } else if x.abs() < tol {
        return x;
    }

    0f64
}

#[allow(dead_code)]
/// The main centering step.  Returns a new version of y so that the entries
/// of u*y that are within tol of {-1, 0, 1} are forced to those values
/// Set tol=0f64 and the function will just return a copy of y
fn center_y( u: &na::DMatrix<f64>, y: &na::DMatrix<f64>, tol: f64 ) -> 
    Option<na::DMatrix<f64>> 
{
    let n = y.nrows();
    let k = y.ncols();

    //Skip if tolerance is zero (i.e. don't center)
    if tol == 0f64 {
        return Some( y.clone() );
    }

    let mut uy = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
    u.mul_to( y, &mut uy );

    //Find epsilon
    let del = uy.map( |x| near_pm_one_or_zero(x, tol) );

    //center y, first we need bfs_inv
    match u.clone().try_inverse() {
        Some(u_inv) => return Some( y - u_inv * del ),
        None => return None,
    }
}


#[allow(dead_code)]
/// Crude test code to test AWGN performance
fn test_awgn() {
    let channels = 50;
    let reps_per = 100;

    let n = 4; let k = 30;
    let complex = false;
    
    //This code does a parameter sweep over the following two variables
    let var = vec![0.001, 0.005, 0.01, 0.05, 0.1]; // Noise variance
    let tol = [0.1];

    //let var = vec![0.008,0.004,0.002,0.001]; // Noise variance
    //let tol = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12]; //Centering tolerance


    let dim = vec![(n, k)];

    let mut res_vec = Vec::new();
    //let mut res_wc_vec = Vec::new();

    for v in 0 .. var.len() {
        for t in 0 .. tol.len() {
        println!("Tolerance: {}", tol[t]);
            eprintln!("Noise variance: {}", var[v]);
            let mut results = TrialResults::new(n,k,var[v]);
            results.tol = tol[t];
            //let mut well_cond_results = TrialResults::new(n,k,var[v]);

            //Generate trial and add noise
            for ii in 0 .. channels {
                if ii % 10 == 0 { eprint!("#"); }

                let mut res = TrialResults::new(n,k,var[v]);
                let x = matrix::get_matrix( &dim[0 .. 1] );
                let (a, y_base) = trial(&x, complex);

                info!("A = {:.4}", a);
                info!("Y = {:.4}", y_base);

                //Mostly for debugging purposes, display the singular values of the
                //channel.  
                if cfg!(build = "debug") {
                    let svd = na::SVD::new(a.clone(), false, false);
                    let s = svd.singular_values;
                    let mut outstr = format!("Singular values: ");
                    let mut min = 1000f64; let mut max = 0f64;
                    for ss in s.iter() {
                        if ss > &max { max = *ss; }
                        if ss < &min { min = *ss; }
                        outstr += &format!("{} ", ss);
                    }
                    trace!("{}", outstr);
                    trace!("Condition number: {}, sigma_4: {}", max / min, min);
                }

                //Main loop
                for _ in 0 .. reps_per {
                    //Generate noise
                    let e = matrix::rand_matrix(n, k);
                    let mut y = y_base.clone() + var[v]*e;
                    res.trials += 1;

                    match single_run(&y,true,tol[t]) {
                        Err(e) => {
                            match e {
                                FlexTabError::Runout => {
                                    res.runout += 1;
                                },
                                
                                _ => res.error += 1,
                            };
                        },

                        Ok(ft) => {
                            res.complete += 1;
                            res.total_bits += n*k;
                            let mut uy = ft.best_state.get_u() * y_base.clone();
                            uy.apply( |x| x.signum() );
                            if equal_atm(&uy, &x) {
                                res.success += 1;
                                info!("EQUAL ATM");
                            } else {
                                res.not_atm += 1;
                                // UY did +not+ match X, print some results and also
                                // determine if UY was even a vertex.
                                info!("Non-ATM");
                                trace!("base_state.uy = {:.2}", ft.best_state.get_uy());
                                trace!("uy = {:.2}", uy );
                                trace!("x = {:.2}", x);
                                let ser = compute_symbol_errors( &uy, &x, 
                                                                 Some(&ft.best_state.get_u()), 
                                                                 Some(&a) );

                                match ser {
                                    Some(s) => {
                                        res.bit_errors += s;
                                    },
                                    //Temporary code until I write better ATM recovery:
                                    None => {
                                        res.bit_errors += 
                                            force_estimate( &uy, &x );
                                    }
                                }
                            }
                        }
                    };
                }
                info!("{}\n", res); 
                
                //if min > 0.1 {
                //    well_cond_results += res.clone();
                //}

                results += res;
            }

            println!("\n{}\n", results);
            //println!("For sigma_4 > 0.1:\n{}\n", well_cond_results);

            res_vec.push( results );
            //res_wc_vec.push( well_cond_results );

        }

        println!("\n-----------------------------------");
    }

    println!("SNR\t delta\t Complete\t BER");
    for i in 0 .. res_vec.len() {
        let n = 1.0 / res_vec[i].var;
        println!("{:.0},\t {:.2},\t {:.2e},\t {:.2e}", 10.0*n.log10(), res_vec[i].tol,
                 res_vec[i].complete as f64 / res_vec[i].trials as f64,
                 res_vec[i].bit_errors as f64 / res_vec[i].total_bits as f64 );
    }
}

#[allow(dead_code)]
//{@
/// Run repetitions of multiple different matrix dimensions, recording success /
/// failure for each matrix types.
//@}
pub fn run_reps() { //{@
    let complex = false;
    let use_basis = true; 
    let reps_per = 100;
    //let dims = vec![(2, 3), (3, 6), (4, 6), (4, 8), (5, 9), (5, 12), (5, 15),
    //    (6, 12), (6, 18), (6, 24), (8, 20), (8, 28), (8, 36)];

    // DEBUG: use this in addition to splice in get_matrix to use static X.
    //let dims = vec![(4, 6)];
    
    //Sweep from n=2 to n=8, skipping n=7 (works but is slow since we don't have an is_done
    //function)
    let mut dims = Vec::new();
    for ii in 2 .. 5 {
        if ii == 7 { continue; }
        for jj in 0 .. 8 {
            if 4*jj <= ii { continue; }
            dims.push( (ii, 4*jj) );
        }
    }
    
    // Setup basic results vector: should migrate this to a full structure.
    // { (n, k), attempts, success, fail = \pm1, fail != \pm1, duration }, 
    //let mut results: Vec<((usize, usize), u64, u64, u64, u64, u64, f64)> = 
    //dims.iter()
    //    .map(|&d| (d, 0, 0, 0, 0, 0, 0.0))
    //    .collect::<Vec<_>>();

    let mut results: Vec<TrialResults> = dims.iter()
        .map(|&d| TrialResults::new(d.0,d.1,0f64))
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
            matrix::get_matrix(&dims[which .. which + 1]) 
        } else {
            matrix::rand_pm1_matrix(dims[which].0, dims[which].1)
        };

        trace!("selected x = {}", x);
        // Get pointer to relevant results tuple for this dimension.
        let ref mut res = results.iter_mut().find(|ref e| e.dims == x.shape()).unwrap();
        res.trials += 1;

        // Obtain A, Y matrices, then run.
        let (a, y) = trial(&x, complex);
        let timer = std::time::Instant::now();
        match single_run(&y,use_basis, 0f64) {
            Err(e) => {
                match e {
                    FlexTabError::Runout => {
                        res.runout += 1; // ran out
                        debug!("ran out of attempts");
                    },
                    _ => {
                        res.error += 1; // something else -- problem
                        println!("critical error = {}", e);
                    },
                };
            },
            Ok(ft) => {
                // Obtained a result: check if UY = X up to an ATM.
                if ft.best_state.uy_equal_atm( &x ) {
                    debug!("EQUAL ATM");
                    res.success += 1;
                } else {
                    // UY did +not+ match X, print some results and also
                    // determine if UY was even a vertex.
                    match a.try_inverse() {
                        None => debug!("UNEQUAL: Cannot take a^-1"),
                        Some(inv) => debug!("UNEQUAL: u = {:.3}a^-1 = {:.3}", 
                                ft.best_state.get_u(), inv),
                    };
                    debug!("UNEXPECTED: return non-ATM");
                    debug!("uy = {:.3}", ft.best_state.get_uy());
                    //if best_state.get_uy().iter().all(|&e|
                    //        (e.abs() - 1.0).abs() < ft.zthresh) {
                    if ft.best_state.uy_is_pm1(ft.get_zthresh()) {
                        res.not_atm += 1; // UY = \pm 1
                    } else {
                        res.error += 1; // UY != \pm 1 -- problem
                        println!("critical error: uy = {:.3}", ft.best_state.get_uy());
                    }
                }
            },
        };
        // Charge elapsed time for this run to its (n, k) dimension.
        let elapsed = timer.elapsed();
        res.time_elapsed += elapsed.as_secs() as f64 + 
                            elapsed.subsec_nanos() as f64 * 1e-9;
    }

    // Print overall results.
    for ref res in results.iter() {
        let mut output = 
            format!("n = {}, k = {:2}: success = {:4} / {:4}, ",
                    (res.dims).0, (res.dims).1, res.success, res.trials);
        output += &format!("(runout = {:2}, pm1 = {:2} err = {:2}), ",
                res.runout, res.not_atm, res.error);
        output += &format!("mean_time_per = {:.5e}", 
                           res.time_elapsed / res.trials as f64);
        println!("{}", output);
    }
} //@}


///Keeps forming problem instances and running the dynamic solver
///until we land on the face of a feasible region.
///Then we try to correct and move to a vertex.
#[allow(dead_code)]
fn single_wrong_dynamic_test( n : usize, k : usize, zthresh : f64 ) 
    -> Option<na::DMatrix<f64>>
{
    loop {
        let dim = vec![(n,k)];
        let x = matrix::get_matrix( &dim[0 .. 1] );
        let (_a, y) = trial( &x, false );

        let u_i = matrix::rand_init(&y);
    
        let mut bfs = match dynamic::find_bfs(&u_i, &y) {
            Some(r) => r, 
            None => return None,
        };

        //Check if we are not in {-1, 0, 1}
        if dynamic::verify_bfs( &bfs, &y, zthresh ) == BFSType::Wrong {
            bfs = match dynamic::find_vertex_on_face( &bfs, &y, zthresh ) {
                Some(r) => r,
                None => return None,
            };
            return Some(bfs * y);
        }
    }
}

#[allow(dead_code)]
fn multiple_dynamic_test( dims: Vec<(usize,usize)>, trials: usize, zthresh: f64 )
{
    for (n, k) in dims.into_iter() {
        let mut pm1 = 0; let mut pm10 = 0; 
        let mut wrong = 0; let mut error = 0;
        println!("Collecting {} trials at n={}, k={}", trials, n, k);
        for i in 0..trials {
            if (i % (trials / 10) == 0) && i != 0 {
                print!(".");
                let _ = stdout().flush();
            }
            let uy = match single_wrong_dynamic_test( n, k, zthresh ) {
                Some(r) => r,
                None => {error += 1; continue;},
            };

            if uy.iter().all( |&elt| (elt.abs() - 1.0).abs() < zthresh ) {
                pm1 += 1;
            } else if uy.iter().all( |&elt| (elt.abs() - 1.0).abs() < zthresh 
                                     || elt.abs() < zthresh ) {
                pm10 += 1;
            } else {
                println!("{:.4}", uy);
                wrong += 1;
            }
        }
        println!("\nPM1: {}, PM10: {}, Wrong: {}, Errors: {}\n", pm1, pm10, wrong, error);
    }
}

//{@
/// Perform a single run using a given +y+ matrix, which contains k symbols each
/// of length n.
//@}r
fn single_run(y: &na::DMatrix<f64>, skip_check: bool, center_tol: f64) 
    -> Result<FlexTab, FlexTabError> 
{ //{@
    let mut attempts = 0;
    const LIMIT: usize = 100; // Max attempts to restart with fresh random U_i.
    let mut ft;
    let mut best: Option<FlexTab> = None;

    //If true, check that y is not singular.
    //Solver should still run if check fails but may be slow and will likely 
    //return garbage
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


    let mut z; //Holds centered version of y
    // Loop trying new u_i until we get n linearly independent \pm 1 cols.
    loop {
        attempts += 1;
        if attempts > LIMIT {
            info!("Ran out of attempts");
            match best {
                Some(b) => return Ok(b),
                None => return Err(FlexTabError::Runout),
            };
        }
        let mut u_i = matrix::rand_init(&y); // Choose rand init start pt.
        //let (y, u_i) = use_given_matrices(); // Use for debugging only.

        z = y.clone();

        //Centering loop
        let mut bfs;
        let mut center_attempts = 0;
        let mut bfs_fail = false;
        loop {
            trace!("y = {:.8}Ui = {:.8}", z, u_i);
            match dynamic::find_bfs(&u_i, &z) {
                Some(b) => bfs = b,
                None => {
                    //This shouldn't really ever happen but does if the input
                    //is poorly conditioned and we run into numerical stability 
                    //issues
                    trace!("Singular starting point, retrying");
                    bfs_fail = true;
                    bfs = u_i;
                    break;
                },
            }

            if (bfs.clone() - u_i.clone()).amax() < ZTHRESH {
                break;
            } else {
                u_i = bfs.clone();
            }

            trace!("bfs = {:.5}", bfs);
            trace!("uy = {:.5}", bfs.clone() * z.clone() );

            //Center y.  If this fails (it never should) then just use
            //the old value of y and try our luck with FlexTab
            z = match center_y( &bfs, &z, center_tol ) {
                Some(yy) => yy,
                None => z,
            };

            trace!("After centering: {:.5}", bfs.clone() * z.clone() );
            center_attempts += 1;
            if center_attempts >= z.shape().0 { break; }
        }

        //If U was singular (can happen if A is ill-conditioned and we are 
        //at the mercy of numerical stability) then try again with new 
        //starting point. Surprisingly we can often recover from this
        if bfs_fail { continue; }

        ft = match FlexTab::new(&bfs, &z, ZTHRESH) {
            Ok(ft) => ft,
            Err(e) => match e {
                 // Insufficient good cols => retry.
                FlexTabError::GoodCols => {
                    warn!("Insufficient good cols, retrying...");
                    continue;
                },
                // Any other error => propogate up.
                _ => return Err(e),
            },
        };

        // Now we have at least n good cols, so try to solve.  If error is that
        // we don't have n linearly independent good cols, then try new u_i.
        // Do same thing if we appear to have been trapped.
        trace!("num_good_cols = {}", ft.num_good_cols());
        debug!("initial ft =\n{}", ft);

        match ft.solve() {
            Ok(_) => break,
            Err(e) => match e {
                FlexTabError::LinIndep 
                    => { 
                        warn!("LinIndep, retrying...");
                    },
                FlexTabError::StateStackExhausted | FlexTabError::TooManyHops
                    => {
                        best = match best {
                            Some(b) => if ft.state.obj() > b.state.obj()
                                { Some(ft) } else { Some(b) },
                            None => Some(ft),
                        };
                        warn!("{}, retrying...", e);
                    },
                _ => return Err(e),
            },
        }
    }

    debug!("FlexTab (n,k) = {:?}: {}, visited = {}", ft.dims(), 
            if ft.has_ybad() { "REDUCED" } else { "FULL" }, ft.visited_vertices());

    // If FlexTab is reduced, we need to do this again starting with a real BFS.
    // Here there is no possibility of insufficient good cols or lin indep cols.
    if ft.has_ybad() {
        debug!("Reduced: now need to solve...");
        let mut ftfull = FlexTab::new(&ft.state.get_u(), &z, ZTHRESH)?;
        match ftfull.solve() {
            Ok(_) => Ok(ftfull),
            Err(e) => {
                println!("ftfull from reduced err = {}", e);
                match e {
                    FlexTabError::StateStackExhausted
                        | FlexTabError::TooManyHops
                        => return Ok(ftfull),
                    _ => return Err(e),
                }
            }
        }
    } else {
        return Ok(ft);
    }
} //@}
// end runnable functions@}

// matrix generation functions{@
#[allow(dead_code)]
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


/// Return true if a is an ATM matrix
/// a is rounded to nearest int so necessary and sufficient condition
/// for a to be an ATM is that each row and column have an l_1 norm of 1
#[allow(dead_code)]
fn is_atm( a: &na::DMatrix<f64> ) -> bool {
    let n = a.shape().0;

    for i in 0 .. n {
        let r = a.row(i).iter()
                        .fold(0, |acc,&e| acc + e.abs() as usize);

        let c = a.column(i).iter()
                           .fold(0, |acc,&e| acc + e.abs() as usize);

        if r != 1 || c != 1 {
            return false;
        }
    }

    true
}

#[allow(dead_code)]
///Find the number of symbol errors in x_hat.  If u, h are provided, then try to 
///recover an ATM.  Otherwise, directly compare x and x_hat
fn compute_symbol_errors( x_hat: &na::DMatrix<f64>, x: &na::DMatrix<f64>,
                         u: Option<&na::DMatrix<f64>>, h: Option<&na::DMatrix<f64>> )
    -> Option<usize>
{
    //Find the error rate of each row
    let (n,_k) = x.shape();
    let mut row_err_vec = vec![0;n];
    for i in 0..n {
        row_err_vec[i] = row_errors( &x_hat, &x, i );
    }

    //If we don't have u and h just return raw error rate
    if u == None || h == None {
        info!("Row errors: {:?}", row_err_vec);
        return Some( row_err_vec.iter().fold(0,|acc,&e| acc + e) );
    }

    //Otherwise, attempt to recover ATM
    match estimate_permutation( x_hat, x, u.unwrap(), h.unwrap() ) {
        Some(p) => {
            let x_hat2 = p * x_hat.clone();

            //Now get raw rate with ATM
            compute_symbol_errors( &x_hat2, x, None, None )      
        },
        None => None,
        //None => Some( row_err_vec.iter().fold(0,|acc,&e| acc + e) ),
    }
}

#[allow(dead_code)]
//Return the number of positons in the ith row where x_hat and x differ by more 
//than 0.5
fn row_errors( x_hat: &na::DMatrix<f64>, x: &na::DMatrix<f64>, i: usize ) -> usize
{
    let x_iter = x.row(i);
    let x_hat_iter = x_hat.row(i);

    //compute raw error rate for this row, error if elements differ by more than 0.5 
    x_iter.iter()
          .enumerate()
          .fold( 0, 
                 |acc, (idx, &e)| 
                 acc + if (e - x_hat_iter[idx]).round() != 0.0 { 1 } else { 0 } 
                )
}


#[allow(dead_code)]
/// Attempt to recover an ATM from U and H.  This is not the most intelligent
/// code but just a starting point.  Often, round(U*H) is all we need to do.
/// If this returns an ATM then we just check if we can flip signs and then
/// stop
fn estimate_permutation( x_hat: &na::DMatrix<f64>, x: &na::DMatrix<f64>,
                         u: &na::DMatrix<f64>, h: &na::DMatrix<f64> )
    -> Option<na::DMatrix<f64>>
{
    let (n,_k) = x.shape();

    //Initial guess at permutation
    let mut p_hat = na::DMatrix::from_column_slice(n, n, &vec![0.0; n*n]);
    u.mul_to(&h, &mut p_hat);
    p_hat.apply( |e| e.round() );
    
    //Check that p_hat is an ATM
    if is_atm( &p_hat ) == false {
        info!("P_hat is not ATM");
        return recover_from_non_atm( &x_hat, &x, &p_hat );
    } else { 
        return None;
        //return recover_from_atm( &x_hat, &x, &p_hat );
    }
}

#[allow(dead_code)]
/// This code just checks whether or not we can improve the BER by
/// fliping signs of each row.
fn recover_from_atm( x_hat: &na::DMatrix<f64>, x: &na::DMatrix<f64>,
                     p_hat: &na::DMatrix<f64>) 
    -> Option<na::DMatrix<f64>>
{
    let (n, k) = x.shape();
    let half = k / 2;
    //let third = k / 3;

    let mut p_hat = p_hat.clone();


    let mut x_tilde = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
    p_hat.mul_to(&x_hat, &mut x_tilde);

    //Get new error vector with permuted rows
    for i in 0..n {
        //Flip sign if bit error rate is over half
        if row_errors( &x_tilde, &x, i) > half {
            let mut row_iter = p_hat.row_mut(i);
            row_iter *= -1f64;
        }
    }
    
    Some(p_hat)
} 

#[allow(dead_code,unused_variables)]
/// This function needs to get written.  In the MATLAB version of the code,
/// if a row has a large error rate, I see if I can do better by permuting rows.
/// For now all this does is start with the identity and flip signs to see if we can
/// do better.  This function doesn't get called too much so fixing this isn't a high
/// priority.
fn recover_from_non_atm( x_hat: &na::DMatrix<f64>, x: &na::DMatrix<f64>,
                         p_hat: &na::DMatrix<f64>) 
    -> Option<na::DMatrix<f64>>
{
    let n = x.shape().0;
    let p_hat = na::DMatrix::<f64>::identity(n,n);

    //TODO: Should add code to try permutations
    recover_from_atm( &x_hat, &x, &p_hat)
}

/// Temp code until I implement the above function.  If estimate_permutation
/// fails, then get a very crude estimate that is at least less than half
fn force_estimate( x_hat: &na::DMatrix<f64>, x: &na::DMatrix<f64> )
    -> usize 
{
    let n = x.shape().0;
    let mut p_hat = na::DMatrix::<f64>::identity(n,n);
    p_hat = recover_from_atm( &x_hat, &x, &p_hat ).unwrap();

    let x_hat2 = p_hat * x_hat.clone();
    compute_symbol_errors( &x_hat2, &x, None, None ).unwrap() 
}
// end matrix generation functions@}

fn trial(x: &na::DMatrix<f64>, complex: bool) -> (na::DMatrix<f64>, na::DMatrix<f64>) { //{@
    trace!("X = {}", x);
    let n = x.nrows();
    let k = x.ncols();

    // Generate random Gaussian matrix A.
    let a = if complex {
        matrix::rand_complex_matrix(n) 
    } else {
        matrix::rand_matrix(n,n)
    };

    // Compute Y = A * X
    let mut y = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
    a.mul_to(&x, &mut y);

    (a, y)
} //@}

fn count_bfs_entry(u: &na::DMatrix<f64>, y: &na::DMatrix<f64>, zthresh: f64) 
    -> (u64,u64,u64) {

    let prod = u*y;
    let pm1 = prod.into_iter().map( |&elt| (elt.abs() - 1.0).abs() < zthresh );
    let zeros = prod.into_iter().map( |&elt| elt.abs() < zthresh );
    let other = prod.into_iter().map( |&elt| ( elt.abs() > zthresh ) 
                                      && ((elt.abs() - 1.0).abs() > zthresh) );

    let sum_pm1 = pm1.into_iter().fold(0, |acc, x| acc + (x as u64));
    let sum_zeros = zeros.into_iter().fold(0, |acc, x| acc + (x as u64));
    let sum_other = other.into_iter().fold(0, |acc, x| acc + (x as u64));

    return (sum_pm1, sum_zeros, sum_other);
}

//{@
/// Returns true if the given value of U is feasible.  Ignores entries in the
/// product UY where mask is set to 0.  This is done to ignore entries that
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

// end principal functions@}
