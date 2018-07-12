extern crate blindsolver;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate nalgebra as na;

use blindsolver::matrix;
use blindsolver::testlib::TrialResults;
use blindsolver::tableau::FlexTabError;

fn main() {
    env_logger::init();

    let complex = false;
    let use_basis = true; 
    let reps_per = 100;
    //let dims = vec![(2, 3), (3, 6), (4, 6), (4, 8), (5, 9), (5, 12), (5, 15),
    //    (6, 12), (6, 18), (6, 24), (8, 20), (8, 28), (8, 36)];

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
        let (a, y) = blindsolver::trial(&x, complex);
        let timer = std::time::Instant::now();
        match blindsolver::single_run(&y,use_basis, 0f64) {
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

/*
fn main() { //{@
    // Initialize logger.
    env_logger::init();

    // Enumeration functions. 
    //neighbor_det_pattern("./n6_equiv.txt");
    //neighbor_det_pattern("./n8_equiv.txt");
    //neighbor_det_pattern_all(5);

    // Run repetitions of the solver.
    run_reps(); //Tests without noise
    //test_awgn(); //Tests with AWGN
    
    //many_bfs(1000);
    
    //let dim = vec![(8,12)];
    //multiple_dynamic_test( dim, 100, ZTHRESH )
    
} //@}
*/
