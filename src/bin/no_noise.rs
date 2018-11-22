extern crate blindsolver;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate nalgebra as na;
extern crate num_cpus;

use blindsolver::matrix;
use blindsolver::testlib::{TrialResults, DimensionSpec};
use std::sync::{Arc, Mutex};
use std::thread;
//use blindsolver::tableau::FlexTabError;

fn main() {
    env_logger::init();

    let num_threads = num_cpus::get();
    eprintln!("Running with {} threads", num_threads);
    let complex = false;
    let use_basis = true; 
    let reps_per = 1000;
    let reps_per_thread = (reps_per / num_threads) as u64;
    let dims = vec![(2, 2, 8), 
                    (3, 3, 13), 
                    (4, 4, 18), 
                    (5, 5, 13), 
                    (6, 6, 22), 
                    (8, 8, 30),
                    (10,10, 100)//,
                    //(12,12, 144)
                    ]; 
    let dims = dims.iter().map(|&(n, m, k)| DimensionSpec::new(n, m, k)).collect::<Vec<_>>();
    let dims = Arc::new(dims);
    //let dims = (0 .. 10).map( |i| DimensionSpec::new(8, 8 + 2*i, 30) )
    //                    .filter(|ref dim| dim.k >= dim.m_rx )
    //                    .collect::<Vec<_>>();


    let results: Vec<TrialResults> = dims.iter()
        .map(|ref d| TrialResults::new(d.n_tx, d.m_rx, d.k, 0f64))
        .collect::<Vec<_>>();
    let results = Arc::new(Mutex::new(results));

    for dim in dims.iter() {
        eprintln!("{}", dim);
        if complex && (dim.n_tx & 1 != 0 || dim.m_rx & 1 != 0) {
            warn!("Complex case must have even n");
            continue;
        }
        if dim.n_tx > dim.m_rx {
            warn!("Must have at least as many receivers as transmitters");
            continue;
        }

        let mut threads = vec![];
        for _ in 0 .. num_threads {
            let dim = dim.clone();
            let results = results.clone();
            let mut t = thread::spawn(move || {
                let mut solver = blindsolver::Solver::new(dim.n_tx, dim.m_rx, 0.0, 100);
                // Select X matrix of one specific set of dimensions (n, k).
                let x = if use_basis {
                    matrix::get_matrix( &[(dim.n_tx, dim.k)] )
                } else {
                    matrix::rand_pm1_matrix(dim.n_tx, dim.k)
                };

                trace!("selected x = {}", x);

                solver.solve_reps(&x, reps_per_thread, complex);
                let mut results = results.lock().unwrap();
                let ref mut res = results.iter_mut().find(|ref e| e.dims == dim).unwrap();
                **res += solver.get_stats();
            });
            threads.push(t);
        }

        for mut t in threads {
            t.join().unwrap();
        }
    }

    // Print overall results.
    let results = results.lock().unwrap();
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

