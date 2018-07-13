extern crate nalgebra as na;
extern crate blindsolver;

use blindsolver::matrix;
use blindsolver::dynamic;

use std::io::stdout;
use std::io::Write;

const ZTHRESH: f64 = 1e-9; 

fn main() {
    let dims: Vec<(usize,usize)> = (5 .. 20).filter(|&x| x % 2 == 0)
                                            .map(|x| (4, x) )
                                            .collect();

    multiple_dynamic_test( dims, 5000 ); 
}

fn single_dynamic_test( n : usize, k : usize ) 
    -> Option<na::DMatrix<f64>>
{
    let dim = vec![(n,k)];
    let x = matrix::get_matrix( &dim[0 .. 1] );
    let (_a, y) = matrix::y_a_from_x( &x, false );

    let u_i = matrix::rand_init(&y);

    let bfs = match dynamic::find_bfs(&u_i, &y) {
        Some(r) => r, 
        None => return None,
    };

    Some(bfs * y)
}

fn multiple_dynamic_test( dims: Vec<(usize,usize)>, trials: usize )
{
    let mut results = vec![(0,0,0,0,0); dims.len()];

    let mut j = 0;
    for (n, k) in dims.clone().into_iter() {
        println!("{} trials at n={}, k={}", trials, n, k);
        for i in 0..trials {
            if (i % (trials / 10) == 0) && i != 0 {
                print!(".");
                let _ = stdout().flush();
            }
            let uy = match single_dynamic_test( n, k ) {
                Some(r) => r,
                None => {results[j].4 += 1; continue;},
            };

            results[j].0 += uy.iter()
                               .filter( |&elt| (elt.abs() - 1.0).abs() < ZTHRESH )
                               .fold( 0, |acc, _x| acc + 1 );
            
            results[j].1 += uy.iter()
                       .filter( |&elt| (elt.abs() - 1.0).abs() < ZTHRESH 
                                     || elt.abs() < ZTHRESH )
                       .fold( 0, |acc, _x| acc + 1 );

            results[j].2 += uy.iter()
                       .filter( |&elt| (elt.abs() - 1.0).abs() > ZTHRESH 
                                     && elt.abs() > ZTHRESH )
                       .fold( 0, |acc, _x| acc + 1 );

            results[j].3 += n*k;
        }
        println!("");
        j = j + 1;
    }

    println!("(n,k)\t\tPM1\t\tPM10\t\tWrong\t\tErrors");
    for i in 0 .. results.len() {
        println!("{:?}\t\t{:.4e}\t{:.4e}\t{:.4e}\t{}",
                 dims[i],
                 (results[i].0 as f64) / (results[i].3 as f64), 
                 (results[i].1 as f64) / (results[i].3 as f64), 
                 (results[i].2 as f64) / (results[i].3 as f64), 
                 results[i].4);
    }
}

