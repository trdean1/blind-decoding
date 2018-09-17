extern crate nalgebra as na;
extern crate blindsolver;
extern crate env_logger;
#[macro_use]
extern crate log;

use blindsolver::matrix;
use blindsolver::testlib::TrialResults;
use blindsolver::tableau::FlexTabError;

/// Crude test code to test AWGN performance
fn main() {
    env_logger::init();

    let channels = 10;
    let reps_per = 100;

    let n_tx = 4;
    let m_rx = 4;
    let k = 30;
    let complex = false;
    
    //This code does a parameter sweep over the following two variables
    let var = vec![0.001]; // Noise variance
    let tol = [0.1];

    //let var = vec![0.008,0.004,0.002,0.001]; // Noise variance
    //let tol = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12]; //Centering tolerance


    let _dim = vec![(n_tx, k)];

    let mut res_vec = Vec::new();
    //let mut res_wc_vec = Vec::new();

    for v in 0 .. var.len() {
        for t in 0 .. tol.len() {
            let mut solver = blindsolver::Solver::new(n_tx, m_rx, tol[t], 100);

            println!("Tolerance: {}", tol[t]);
            eprintln!("Noise variance: {}", var[v]);
            let mut results = TrialResults::new(n_tx, k, var[v]);
            results.tol = tol[t];
            //let mut well_cond_results = TrialResults::new(n,k,var[v]);

            //Generate trial and add noise
            for ii in 0 .. channels {
                if ii % 10 == 0 && ii != 0 { eprint!("#"); }

                let mut res = TrialResults::new(n_tx, k, var[v]);
                let x = matrix::get_matrix(&[(n_tx, k)]);
                let (a, y_base) = matrix::y_a_from_x(&x, n_tx, complex);
                let y_reduced = match matrix::rank_reduce(&y_base, n_tx) {
                    Some(y) => y,
                    None => { res.error += 1; continue; }
                };


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
                    let e = matrix::rand_matrix(n_tx, k);
                    let mut y = y_base.clone() + var[v]*e;
                    res.trials += 1;

                    match solver.solve( &y ) {
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
                            res.total_bits += n_tx * k;
                            //let mut uy = ft.state.get_u() * y_base.clone();
                            let mut uy = ft.state.get_u() * y_reduced.clone();
                            uy.apply( |x| x.signum() );
                            if blindsolver::equal_atm(&uy, &x) {
                                res.success += 1;
                                info!("EQUAL ATM");
                            } else {
                                res.not_atm += 1;
                                // UY did +not+ match X, print some results and also
                                // determine if UY was even a vertex.
                                info!("Non-ATM");
                                trace!("base_state.uy = {:.2}", ft.state.get_uy());
                                trace!("uy = {:.2}", uy );
                                trace!("x = {:.2}", x);
                                let ser = blindsolver::compute_symbol_errors( &uy, &x, 
                                                                              Some(&ft.state.get_u()), 
                                                                              Some(&a) );

                                match ser {
                                    Some(s) => {
                                        res.bit_errors += s;
                                    },
                                    //Temporary code until I write better ATM recovery:
                                    None => {
                                        res.bit_errors += 
                                            blindsolver::force_estimate( &uy, &x );
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
