extern crate blindsolver;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate nalgebra as na;

use blindsolver::matrix;
use blindsolver::testlib::TrialResults;
//use blindsolver::tableau::FlexTabError;

fn main() {
    env_logger::init();

    let complex = false;
    let use_basis = true; 
    let reps_per = 1000;
    //let dims = vec![(2,8)];
    let dims = vec![(2, 8), (3, 13), (4, 18), (5, 13), (6, 22), (8, 30)];
    //let dims = vec![(8,30)];
    //let dims = vec![(10,45)];

    //Sweep from n=2 to n=8, skipping n=7 (works but is slow since we don't have an is_done
    //function)
    
    /*
    let mut dims = Vec::new();
    for ii in 2 .. 9 {
        if ii == 7 { continue; }
        for jj in 0 .. 9 {
            if 4*jj <= ii { continue; }
            dims.push( (ii, 4*jj) );
        }
    }*/

    let mut results: Vec<TrialResults> = dims.iter()
        .map(|&d| TrialResults::new(d.0,d.1,0f64))
        .collect::<Vec<_>>();

    for &dim in dims.iter() {
        eprintln!("Dim = {:?}", dim);
        if complex && (dim.0 % 2 != 0) {
            warn!("Complex case must have even n");
            continue;
        }
        let mut solver = blindsolver::Solver::new( dim.0, dim.0 + 2, false, 0.0, 100 );
        // Select X matrix of one specific set of dimensions (n, k).
        let x = if use_basis {
            matrix::get_matrix(&[dim])
        } else {
            matrix::rand_pm1_matrix(dim.0, dim.1)
        };

        trace!("selected x = {}", x);
        // Get pointer to relevant results tuple for this dimension.
        let ref mut res = results.iter_mut().find(|ref e| e.dims == x.shape()).unwrap();

        for _iter in 0 .. reps_per {
            if reps_per > 10 {
                if _iter % (reps_per / 10) == 0 {
                    eprint!("#");
                } else if _iter % reps_per == reps_per - 1 {
                    eprint!("\n");
                }
            }

            // Obtain A, Y matrices, then run.
            let (_a, y) = matrix::y_a_from_x(&x, complex);
            debug!("Y = {:.02}", y);
            //let timer = std::time::Instant::now();
            match solver.solve( &y ) {
                Err(_) => {
                    /*match e {
                        FlexTabError::Runout => {
                            res.runout += 1; // ran out
                            debug!("ran out of attempts");
                        },
                        _ => {
                            res.error += 1; // something else -- problem
                            println!("critical error = {}", e);
                        },
                    };*/
                },
                Ok(ft) => {
                    // Obtained a result: check if UY = X up to an ATM.
                    let u = ft.state.get_u();
                    let uy = u * y.clone();
                    if blindsolver::equal_atm( &uy, &x ) {
                        res.success += 1;
                    } else {
                        // UY did +not+ match X, print some results and also
                        // determine if UY was even a vertex.
                        //match a.try_inverse() {
                        //    None => debug!("UNEQUAL: Cannot take a^-1"),
                        //    Some(inv) => debug!("UNEQUAL: u = {:.3}a^-1 = {:.3}", 
                        //            ft.best_state.get_u(), inv),
                        //};
                        if blindsolver::is_pm1( &uy, ft.get_zthresh() ) {
                            res.not_atm += 1; // UY = \pm 1
                        } else {
                            res.error += 1; // UY != \pm 1 -- problem
                            debug!("critical error: uy = {:.3}", uy );
                        }
                    }
                },
            };
        }
        **res += solver.get_stats();
    }

    // Print overall results.
    for ref res in results.iter() {
        let mut output = format!("n = {}, k = {:2}: success = {:4} / {:4}, ",
                (res.dims).0, (res.dims).1, res.success, res.trials);
        output += &format!("(runout = {:2}, pm1 = {:2}, err = {:2}), ",
                res.runout, res.not_atm, res.error);
        output += &format!("mean_time_per = {:.5e}", res.time_elapsed / res.trials as f64);
        println!("{}", output);
    }

    /*
    println!("\nPer k:");
    for ref res in results.iter() {
        println!("({},{:.02e})", (res.dims).1, (res.time_elapsed / res.trials as f64) / (res.dims).1 as f64 ); 
    }*/
} 
