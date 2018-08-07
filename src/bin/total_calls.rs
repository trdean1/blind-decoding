extern crate blindsolver;
extern crate nalgebra as na;

use blindsolver::matrix;

fn main() {
    let reps_per = 10000;
    let dims = vec![(2,8)];

    let mut solver = blindsolver::Solver::new( false, 0.0, 100 );

    let mut bfs_total = 0;
    let mut ft_total = 0;
    let mut success = 0;

    println!("Dim = {:?}", dims[0]);
    for iter in 0 .. reps_per {
        solver.clear_stats();

        if iter != 0 && (iter % (reps_per / 10) == 0) {
            eprint!("#");
        } else if iter % reps_per == reps_per - 1 {
            eprint!("\n");
        }

        // Select X matrix of one specific set of dimensions (n, k).
        let x  = matrix::get_matrix(&dims[0 .. 1]); 

        // Obtain A, Y matrices, then run.
        let (_a, y) = matrix::y_a_from_x(&x, false);

        match solver.single_run( &y ) {
            Err(_) => {},
            Ok(ft) => {
                if ft.best_state.uy_equal_atm( &x ) {
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
