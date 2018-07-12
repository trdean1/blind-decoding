extern crate blindsolver;
extern crate env_logger;

fn main() {
    env_logger::init();
    blindsolver::run_reps();
}

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
