extern crate blindsolver;
extern crate nalgebra as na;

use blindsolver::matrix;
use blindsolver::testlib::DimensionSpec;

fn main() {
    let reps_per = 1000;
    let dims = vec![(8, 8, 30)];
    let dims = dims.iter().map(|&(n, m, k)| DimensionSpec::new(n, m, k)).collect::<Vec<_>>();

    let mut bfs_total = 0;
    let mut ft_total = 0;
    let mut success = 0;

    for ref dim in dims.iter() {
        let mut solver = blindsolver::Solver::new(dim.n_tx, dim.m_rx, 0.0, 100);
        println!("{}", dim);
        for iter in 0 .. reps_per {
            if iter != 0 && (iter % (reps_per / 10) == 0) {
                eprint!("#");
            } else if iter % reps_per == reps_per - 1 {
                eprint!("\n");
            }
    
            // Select X matrix of one specific set of dimensions (n, k).
            let x  = matrix::get_matrix( &[(dim.n_tx, dim.k)] ); 
    
            // Obtain A, Y matrices, then run.
            let (_a, y) = matrix::y_a_from_x(&x, dim.m_rx, false);
            let y_reduced = match matrix::rank_reduce(&y, dim.n_tx) {
                Some(y) => y,
                None => { /* res.error += 1; */ continue; }
            };
    
            match solver.solve( &y ) {
                Err(_) => {},
                Ok(ft) => {
                    let u = ft.state.get_u();
                    //let uy = u * y.clone();
                    let uy = u * y_reduced.clone();
                    if blindsolver::equal_atm( &uy, &x ) {
                        success += 1;
                    }
                },
            };
            let new_res = solver.get_stats();
    
            bfs_total += 1 + new_res.goodcols;
            bfs_total += new_res.linindep;
            bfs_total += new_res.statestack;
            bfs_total += new_res.toomanyhops;
            bfs_total += new_res.trap;
    
            ft_total += 1 + new_res.statestack;
            ft_total += new_res.toomanyhops;
            ft_total += new_res.trap;
            ft_total += new_res.reduced;
        }
    
        println!("Success: {} / {}", success, reps_per);
        println!("Total BFS calls: {}", bfs_total);
        println!("Total FT calls: {}", ft_total);
    }
}
