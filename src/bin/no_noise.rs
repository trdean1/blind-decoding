extern crate blindsolver;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate nalgebra as na;

use blindsolver::matrix;
use blindsolver::testlib::{TrialResults, DimensionSpec};
//use blindsolver::tableau::FlexTabError;

fn main() {
    env_logger::init();

    let complex = false;
    let use_basis = true; 
    let reps_per = 1000;
    let dims = vec![(2, 2, 8), 
                    (3, 3, 13), 
                    (4, 4, 18), 
                    (5, 5, 13), 
                    (6, 6, 22), 
                    (8, 8, 30),
                    (10,10, 100)]; 
    let dims = dims.iter().map(|&(n, m, k)| DimensionSpec::new(n, m, k)).collect::<Vec<_>>();
    //let dims = (0 .. 10).map( |i| DimensionSpec::new(8, 8 + 2*i, 30) )
    //                    .filter(|ref dim| dim.k >= dim.m_rx )
    //                    .collect::<Vec<_>>();

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
        .map(|ref d| TrialResults::new(d.n_tx, d.m_rx, d.k, 0f64))
        .collect::<Vec<_>>();

    for ref dim in dims.iter() {
        eprintln!("{}", dim);
        if complex && (dim.n_tx & 1 != 0 || dim.m_rx & 1 != 0) {
            warn!("Complex case must have even n");
            continue;
        }
        if dim.n_tx > dim.m_rx {
            warn!("Must have at least as many receivers as transmitters");
            continue;
        }

        let mut solver = blindsolver::Solver::new(dim.n_tx, dim.m_rx, 0.0, 100);
        // Select X matrix of one specific set of dimensions (n, k).
        let x = if use_basis {
            matrix::get_matrix( &[(dim.n_tx, dim.k)] )
        } else {
            matrix::rand_pm1_matrix(dim.n_tx, dim.k)
        };

        trace!("selected x = {}", x);
        // Get pointer to relevant results tuple for this dimension.
        let ref mut res = results.iter_mut().find(|ref e| e.dims == **dim).unwrap();

        for _iter in 0 .. reps_per {
            if reps_per > 10 {
                if _iter % (reps_per / 10) == 0 {
                    eprint!("#");
                } else if _iter % reps_per == reps_per - 1 {
                    eprint!("\n");
                }
            }

            // Obtain A, Y matrices, then run.
            let (_a, y) = matrix::y_a_from_x(&x, dim.m_rx, complex);
            let y_reduced = match matrix::rank_reduce(&y, dim.n_tx) {
                Some(y) => y,
                None => { res.error += 1; continue; }
            };
            
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
                    let uy = u * y_reduced.clone();
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
        let mut output = format!("{}: success = {:4} / {:4}, ",
                res.dims, res.success, res.trials);
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
