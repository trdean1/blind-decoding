extern crate blindsolver;
extern crate nalgebra as na;

use blindsolver::matrix;

fn main() {
    let reps_per = 1000;
    let dims = vec![(8,30)];

    let mut bfs_total = 0;
    let mut ft_total = 0;
    let mut success = 0;

    for &dim in dims.iter() {
        let mut solver = blindsolver::Solver::new( dim.0, dim.0, false, 0.0, 100 );
        println!("Dim = {:?}", dim);
        for iter in 0 .. reps_per {
    
            if iter != 0 && (iter % (reps_per / 10) == 0) {
                eprint!("#");
            } else if iter % reps_per == reps_per - 1 {
                eprint!("\n");
            }
    
            // Select X matrix of one specific set of dimensions (n, k).
            let x  = matrix::get_matrix(&[dim]); 
    
            // Obtain A, Y matrices, then run.
            let (_a, y) = matrix::y_a_from_x(&x, false);
    
            match solver.solve( &y ) {
                Err(_) => {},
                Ok(ft) => {
                    let u = ft.state.get_u();
                    let uy = u * y.clone();
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
