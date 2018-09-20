extern crate blindsolver;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate nalgebra as na;

use blindsolver::matrix;
use blindsolver::testlib::TrialResults;

fn main() {
    env_logger::init();

    let reps = 1000;
    let n = 12;

    let mut results = TrialResults::new(n, n, n, 0f64);
    let mut solver = blindsolver::Solver::new(n, n, 0.0, 100);

    let x = matrix::get_matrix( &[(n, n)] );
    let x_det = x.determinant().abs();

    for iter in 0 .. reps {

        trace!("selected x = {}", x);
        // Get pointer to relevant results tuple for this dimension.

        if reps > 10 {
            if iter % (reps / 10) == 0 {
                eprint!("#");
            } else if iter % reps == reps - 1 {
                eprint!("\n");
            }
        }

        // Obtain A, Y matrices, then run.
        let (_a, y) = matrix::ortho_y_a_from_x(&x);
            
        debug!("Y = {:.02}", y);
        match solver.solve( &y ) {
            Err(_) => {

            },
            Ok(mut ft) => {
                // Obtained a result: check if UY = X up to an ATM.
                //ft.restore_best_state();
                let u = ft.state.get_u();
                let uy = u * y.clone();
                let found_det = uy.determinant().abs();

                /*
                println!("{}", ft.print_flip_grad());
                println!("uy = {:.03}", uy);
                println!("Found: {:.01} vs Start: {:.01}\n", found_det, x_det);
                println!("Best: {:.03} vs Current: {:.03}\n", ft.best_obj, ft.state.obj());
                println!("--------------------------------------------\n");
                */

                if (found_det - x_det).abs() < 1.0 {
                    results.success += 1;
                } else {
                    println!("{}", ft.print_flip_grad());
                    println!("uy = {:.03}", uy);
                    println!("Found: {:.01} vs Start: {:.01}\n", found_det, x_det);
                    println!("Best: {:.03} vs Current: {:.03}\n", ft.best_obj, ft.state.obj());
                    println!("--------------------------------------------\n");
                }

                if blindsolver::is_pm1( &uy, ft.get_zthresh() ) {
                    results.not_atm += 1; // UY = \pm 1
                } 
                
            },
        };
    }

    results  += solver.get_stats();

    let mut output = format!("{}: success = {:4} / {:4}, ",
            results.dims, results.success, results.trials);
    output += &format!("(runout = {:2}, pm1 = {:2}) ",
            results.runout, results.not_atm);
    output += &format!("mean_time_per = {:.5e}", results.time_elapsed / results.trials as f64);
    println!("{}", output);
}
