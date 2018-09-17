extern crate blindsolver;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate nalgebra as na;

use blindsolver::matrix;
use blindsolver::testlib::{TrialResults, DimensionSpec};

use std::collections::HashMap; 

fn main() {
    env_logger::init();

    let reps_per = 1000;
    let dims = vec![(8, 8, 30)];
    let dims = dims.iter().map(|&(n, m, k)| DimensionSpec::new(n, m, k)).collect::<Vec<_>>();

    let mut bfs_histogram: HashMap<usize, usize> = HashMap::new(); 
    let mut linindep_histogram: HashMap<usize, usize> = HashMap::new(); 
    let mut statestack_histogram: HashMap<usize, usize> = HashMap::new(); 
    let mut hop_histogram: HashMap<usize, usize> = HashMap::new(); 
    let mut trap_histogram: HashMap<usize, usize> = HashMap::new(); 
    
    let mut results = TrialResults::new(dims[0].n_tx, dims[0].k, 0f64);

    for ref dim in dims.iter() {
        println!("{}", dim);
        let mut solver = blindsolver::Solver::new(dim.n_tx, dim.m_rx, 0.0, 100);
        for iter in 0 .. reps_per {
            debug!("\n\n--------------------------");
    
            if iter != 0 && (iter % (reps_per / 10) == 0) {
                eprint!("#");
            } else if iter % reps_per == reps_per - 1 {
                eprint!("\n");
            }
    
            // Select X matrix of one specific set of dimensions (n, k).
            let x  = matrix::get_matrix( &[(dim.n_tx, dim.k)] ); 
            debug!("X = {}", x);
    
            // Obtain A, Y matrices, then run.
            let (_a, y) = matrix::y_a_from_x(&x, dim.m_rx, false);
            let y_reduced = match matrix::rank_reduce(&y, dim.n_tx) {
                Some(y) => y,
                None => { results.error += 1; continue; }
            };
            debug!("Y = {}", y);
    
            match solver.solve( &y ) {
                Err(_) => {},
                Ok(ft) => {
                    // Obtained a result: check if UY = X up to an ATM.
                    let u = ft.state.get_u();
                    //let uy = u * y.clone();
                    let uy = u * y_reduced.clone();
                    if blindsolver::equal_atm( &uy, &x ) {
                        results.success += 1;
                    } else {
                        // UY did +not+ match X, print some results and also
                        // determine if UY was even a vertex.
                        //match a.try_inverse() {
                        //    None => debug!("UNEQUAL: Cannot take a^-1"),
                        //    Some(inv) => debug!("UNEQUAL: u = {:.3}a^-1 = {:.3}", 
                        //            ft.best_state.get_u(), inv),
                        //};
                        if blindsolver::is_pm1( &uy, ft.get_zthresh() ) {
                            results.not_atm += 1; // UY = \pm 1
                        } else {
                            results.error += 1; // UY != \pm 1 -- problem
                            debug!("critical error: uy = {:.3}", uy );
                        }
                    }
                },
            };
            let new_res = solver.get_stats();
            results += new_res.clone();
    
            let bfs_stat = bfs_histogram.entry(new_res.goodcols).or_insert(0);
            let linindep_stat = linindep_histogram.entry(new_res.linindep).or_insert(0);
            let statestack_stat = statestack_histogram.entry(new_res.statestack).or_insert(0);
            let hop_stat = hop_histogram.entry(new_res.toomanyhops).or_insert(0);
            let trap_stat = trap_histogram.entry(new_res.trap).or_insert(0);
            *bfs_stat += 1;
            *linindep_stat += 1;
            *statestack_stat += 1;
            *hop_stat += 1;
            *trap_stat += 1;
        }
    }

    // Print overall results.
    let mut output = 
        format!("n = {}, k = {:2}: success = {:4} / {:4}, ",
                (results.dims).0, (results.dims).1, results.success, results.trials);

    output += &format!("(runout = {:2}, pm1 = {:2}, trap={:2}, err = {:2}), ",
            results.runout, results.not_atm, results.trap, results.error);

    println!("{}", output);
    results.extended_results();
    println!("");
    println!("Insufficient good columns:");
    pp_hash( &bfs_histogram );
    println!("Too few independent columns:");
    pp_hash( &linindep_histogram );
    println!("State Stack Exhausted:");
    pp_hash( &statestack_histogram );
    println!("Too many hops:");
    pp_hash( &hop_histogram );
    println!("Trapped:");
    pp_hash( &trap_histogram );
} 

fn pp_hash( hm: &HashMap<usize, usize> ) {
    let mut pairs: Vec<(usize,usize)> = Vec::new();

    for (k,v) in hm.into_iter() {
        pairs.push( (*k, *v) );
    }

    pairs.sort_by( |(a,_b), (c,_d)| a.cmp(c) );

    println!("Runs\tFrequency");
    println!("-----------------");
    for (k,v) in pairs.iter() {
        println!("{}:\t{}", k,v);
    }
    println!("\n");
}
