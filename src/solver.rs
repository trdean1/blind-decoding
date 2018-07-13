/*
 * This code is proprietary information and property of Stanford University.
 * Do not share without permission of the authors.  This code may only be
 * used for educational purposes.  
 *
 * Authors: Jonathan Perlstein (jrperl@cs.stanford.edu)
 *          Thomas Dean (trdean@stanford.edu)
 *
 */

extern crate nalgebra as na;
extern crate rand;
extern crate regex;
#[macro_use]
extern crate log;
extern crate env_logger;

pub mod matrix;
pub mod testlib;
pub mod tableau;
pub mod dynamic;

use tableau::FlexTabError;
use tableau::FlexTab;

const ZTHRESH: f64 = 1e-9;

/// Perform a single run using a given +y+ matrix, which contains k symbols each
/// of length n.
pub fn single_run(y: &na::DMatrix<f64>, skip_check: bool, center_tol: f64) 
    -> Result<FlexTab, FlexTabError> 
{ 
    let mut attempts = 0;
    const LIMIT: usize = 100; // Max attempts to restart with fresh random U_i.
    let mut ft;
    let mut best: Option<FlexTab> = None;

    //If true, check that y is not singular.
    //Solver should still run if check fails but may be slow and will likely 
    //return garbage
    if skip_check == false {
       let (n,_k) = y.shape();
       let svd = na::SVD::new(y.clone(), false, false);
       let s = svd.singular_values;
       trace!("Singular values =\n");
       for ss in s.iter() {
           trace!("{}", ss);
       }
       let r = s.iter().filter(|&elt| *elt > ZTHRESH).count();
       trace!("({})\n",r);
       if r < n {
            return Err(FlexTabError::SingularInput);
       }
    }


    let mut z; //Holds centered version of y
    // Loop trying new u_i until we get n linearly independent \pm 1 cols.
    loop {
        attempts += 1;
        if attempts > LIMIT {
            info!("Ran out of attempts");
            match best {
                Some(b) => return Ok(b),
                None => return Err(FlexTabError::Runout),
            };
        }
        let mut u_i = matrix::rand_init(&y); // Choose rand init start pt.
        //let (y, u_i) = use_given_matrices(); // Use for debugging only.

        z = y.clone();

        //Centering loop
        let mut bfs;
        let mut center_attempts = 0;
        let mut bfs_fail = false;
        loop {
            trace!("y = {:.8}Ui = {:.8}", z, u_i);
            match dynamic::find_bfs(&u_i, &z) {
                Some(b) => bfs = b,
                None => {
                    //This shouldn't really ever happen but does if the input
                    //is poorly conditioned and we run into numerical stability 
                    //issues
                    trace!("Singular starting point, retrying");
                    bfs_fail = true;
                    bfs = u_i;
                    break;
                },
            }

            if center_tol == 0f64 { break; }

            if (bfs.clone() - u_i.clone()).amax() < ZTHRESH {
                break;
            } else {
                u_i = bfs.clone();
            }

            trace!("bfs = {:.5}", bfs);
            trace!("uy = {:.5}", bfs.clone() * z.clone() );

            //Center y.  If this fails (it never should) then just use
            //the old value of y and try our luck with FlexTab
            z = match center_y( &bfs, &z, center_tol ) {
                Some(yy) => yy,
                None => z,
            };

            trace!("After centering: {:.5}", bfs.clone() * z.clone() );
            center_attempts += 1;
            if center_attempts >= z.shape().0 { break; }
        }

        //If U was singular (can happen if A is ill-conditioned and we are 
        //at the mercy of numerical stability) then try again with new 
        //starting point. Surprisingly we can often recover from this
        if bfs_fail { continue; }

        ft = match FlexTab::new(&bfs, &z, ZTHRESH) {
            Ok(ft) => ft,
            Err(e) => match e {
                 // Insufficient good cols => retry.
                FlexTabError::GoodCols => {
                    warn!("Insufficient good cols, retrying...");
                    continue;
                },
                // Any other error => propogate up.
                _ => return Err(e),
            },
        };

        // Now we have at least n good cols, so try to solve.  If error is that
        // we don't have n linearly independent good cols, then try new u_i.
        // Do same thing if we appear to have been trapped.
        trace!("num_good_cols = {}", ft.num_good_cols());
        debug!("initial ft =\n{}", ft);

        match ft.solve() {
            Ok(_) => break,
            Err(e) => match e {
                FlexTabError::LinIndep 
                    => { 
                        warn!("LinIndep, retrying...");
                    },
                FlexTabError::StateStackExhausted | FlexTabError::TooManyHops
                    => {
                        best = match best {
                            Some(b) => if ft.state.obj() > b.state.obj()
                                { Some(ft) } else { Some(b) },
                            None => Some(ft),
                        };
                        warn!("{}, retrying...", e);
                    },
                _ => return Err(e),
            },
        }
    }

    debug!("FlexTab (n,k) = {:?}: {}, visited = {}", ft.dims(), 
            if ft.has_ybad() { "REDUCED" } else { "FULL" }, ft.visited_vertices());

    // If FlexTab is reduced, we need to do this again starting with a real BFS.
    // Here there is no possibility of insufficient good cols or lin indep cols.
    if ft.has_ybad() {
        debug!("Reduced: now need to solve...");
        let mut ftfull = FlexTab::new(&ft.state.get_u(), &z, ZTHRESH)?;
        match ftfull.solve() {
            Ok(_) => Ok(ftfull),
            Err(e) => {
                println!("ftfull from reduced err = {}", e);
                match e {
                    FlexTabError::StateStackExhausted
                        | FlexTabError::TooManyHops
                        => return Ok(ftfull),
                    _ => return Err(e),
                }
            }
        }
    } else {
        return Ok(ft);
    }
}

#[allow(dead_code)]
/// Return true iff a == b up to an ATM.
pub fn equal_atm(a: &na::DMatrix<f64>, b: &na::DMatrix<f64>) -> bool { //{@
    if a.shape() != b.shape() { return false; }
    // used_rows[k] will be set to true when a row k of +b+ is used to match
    // some row in +a+.
    let mut used_rows = vec![false; b.shape().0];

    // Iterate over all rows of +a+; foreach row find a (previously unmatched)
    // row of +b+ that matches this row of +a+ up to a sign inversion.
    for i in 0 .. a.shape().0 {
        let mut matched = false;
        for k in 0 .. used_rows.len() {
            if used_rows[k] { continue; }
            // Calc difference of vector a[i] - b[i].
            if (a.row(i) - b.row(k)).amax() < ZTHRESH
                    || (a.row(i) + b.row(k)).amax() < ZTHRESH {
                matched = true;
                used_rows[k] = true;
            }
        }
        if !matched { return false; }
    }
    true
} 


#[allow(dead_code)]
fn count_bfs_entry(u: &na::DMatrix<f64>, y: &na::DMatrix<f64>, zthresh: f64) 
    -> (u64,u64,u64) {

    let prod = u*y;
    let pm1 = prod.into_iter().map( |&elt| (elt.abs() - 1.0).abs() < zthresh );
    let zeros = prod.into_iter().map( |&elt| elt.abs() < zthresh );
    let other = prod.into_iter().map( |&elt| ( elt.abs() > zthresh ) 
                                      && ((elt.abs() - 1.0).abs() > zthresh) );

    let sum_pm1 = pm1.into_iter().fold(0, |acc, x| acc + (x as u64));
    let sum_zeros = zeros.into_iter().fold(0, |acc, x| acc + (x as u64));
    let sum_other = other.into_iter().fold(0, |acc, x| acc + (x as u64));

    return (sum_pm1, sum_zeros, sum_other);
}

/// Returns true if the given value of U is feasible.  Ignores entries in the
/// product UY where mask is set to 0.  This is done to ignore entries that
fn is_feasible(u: &na::DMatrix<f64>, y: &na::DMatrix<f64>,
               mask: Option<&na::DMatrix<bool>>) -> bool {
    let prod = u * y;
    if let Some(mask) = mask {
        assert!(prod.shape() == mask.shape());
    }

    for j in 0 .. prod.ncols() {
        for i in 0 .. prod.nrows() {
            let check = match mask {
                None => true,
                Some(mask) => mask.column(j)[i],
            };
            if check && prod.column(j)[i].abs() > 1.0f64 + ZTHRESH {
                return false;
            }
        }
    }
    true
}

/////////////////////////////////////////////////////////////////////////
///
/// This is all code for dealing with AWGN / bit errors.  Needs cleaning
/// up and is somewhat incomplete.  Will likely move to a its own module
/// eventually.
///
/////////////////////////////////////////////////////////////////////////

/// Return true if a is an ATM matrix
/// a is rounded to nearest int so necessary and sufficient condition
/// for a to be an ATM is that each row and column have an l_1 norm of 1
#[allow(dead_code)]
pub fn is_atm( a: &na::DMatrix<f64> ) -> bool {
    let n = a.shape().0;

    for i in 0 .. n {
        let r = a.row(i).iter()
                        .fold(0, |acc,&e| acc + e.abs() as usize);

        let c = a.column(i).iter()
                           .fold(0, |acc,&e| acc + e.abs() as usize);

        if r != 1 || c != 1 {
            return false;
        }
    }

    true
}

#[allow(dead_code)]
///Find the number of symbol errors in x_hat.  If u, h are provided, then try to 
///recover an ATM.  Otherwise, directly compare x and x_hat
pub fn compute_symbol_errors( x_hat: &na::DMatrix<f64>, x: &na::DMatrix<f64>,
                         u: Option<&na::DMatrix<f64>>, h: Option<&na::DMatrix<f64>> )
    -> Option<usize>
{
    //Find the error rate of each row
    let (n,_k) = x.shape();
    let mut row_err_vec = vec![0;n];
    for i in 0..n {
        row_err_vec[i] = row_errors( &x_hat, &x, i );
    }

    //If we don't have u and h just return raw error rate
    if u == None || h == None {
        info!("Row errors: {:?}", row_err_vec);
        return Some( row_err_vec.iter().fold(0,|acc,&e| acc + e) );
    }

    //Otherwise, attempt to recover ATM
    match estimate_permutation( x_hat, x, u.unwrap(), h.unwrap() ) {
        Some(p) => {
            let x_hat2 = p * x_hat.clone();

            //Now get raw rate with ATM
            compute_symbol_errors( &x_hat2, x, None, None )      
        },
        None => None,
        //None => Some( row_err_vec.iter().fold(0,|acc,&e| acc + e) ),
    }
}

#[allow(dead_code)]
//Return the number of positons in the ith row where x_hat and x differ by more 
//than 0.5
fn row_errors( x_hat: &na::DMatrix<f64>, x: &na::DMatrix<f64>, i: usize ) -> usize
{
    let x_iter = x.row(i);
    let x_hat_iter = x_hat.row(i);

    //compute raw error rate for this row, error if elements differ by more than 0.5 
    x_iter.iter()
          .enumerate()
          .fold( 0, 
                 |acc, (idx, &e)| 
                 acc + if (e - x_hat_iter[idx]).round() != 0.0 { 1 } else { 0 } 
                )
}


#[allow(dead_code)]
/// Attempt to recover an ATM from U and H.  This is not the most intelligent
/// code but just a starting point.  Often, round(U*H) is all we need to do.
/// If this returns an ATM then we just check if we can flip signs and then
/// stop
fn estimate_permutation( x_hat: &na::DMatrix<f64>, x: &na::DMatrix<f64>,
                         u: &na::DMatrix<f64>, h: &na::DMatrix<f64> )
    -> Option<na::DMatrix<f64>>
{
    let (n,_k) = x.shape();

    //Initial guess at permutation
    let mut p_hat = na::DMatrix::from_column_slice(n, n, &vec![0.0; n*n]);
    u.mul_to(&h, &mut p_hat);
    p_hat.apply( |e| e.round() );
    
    //Check that p_hat is an ATM
    if is_atm( &p_hat ) == false {
        info!("P_hat is not ATM");
        return recover_from_non_atm( &x_hat, &x, &p_hat );
    } else { 
        return None;
        //return recover_from_atm( &x_hat, &x, &p_hat );
    }
}

#[allow(dead_code)]
/// This code just checks whether or not we can improve the BER by
/// fliping signs of each row.
fn recover_from_atm( x_hat: &na::DMatrix<f64>, x: &na::DMatrix<f64>,
                     p_hat: &na::DMatrix<f64>) 
    -> Option<na::DMatrix<f64>>
{
    let (n, k) = x.shape();
    let half = k / 2;
    //let third = k / 3;

    let mut p_hat = p_hat.clone();


    let mut x_tilde = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
    p_hat.mul_to(&x_hat, &mut x_tilde);

    //Get new error vector with permuted rows
    for i in 0..n {
        //Flip sign if bit error rate is over half
        if row_errors( &x_tilde, &x, i) > half {
            let mut row_iter = p_hat.row_mut(i);
            row_iter *= -1f64;
        }
    }
    
    Some(p_hat)
} 

#[allow(dead_code,unused_variables)]
/// This function needs to get written.  In the MATLAB version of the code,
/// if a row has a large error rate, I see if I can do better by permuting rows.
/// For now all this does is start with the identity and flip signs to see if we can
/// do better.  This function doesn't get called too much so fixing this isn't a high
/// priority.
fn recover_from_non_atm( x_hat: &na::DMatrix<f64>, x: &na::DMatrix<f64>,
                         p_hat: &na::DMatrix<f64>) 
    -> Option<na::DMatrix<f64>>
{
    let n = x.shape().0;
    let p_hat = na::DMatrix::<f64>::identity(n,n);

    //TODO: Should add code to try permutations
    recover_from_atm( &x_hat, &x, &p_hat)
}

/// Temp code until I implement the above function.  If estimate_permutation
/// fails, then get a very crude estimate that is at least less than half
pub fn force_estimate( x_hat: &na::DMatrix<f64>, x: &na::DMatrix<f64> )
    -> usize 
{
    let n = x.shape().0;
    let mut p_hat = na::DMatrix::<f64>::identity(n,n);
    p_hat = recover_from_atm( &x_hat, &x, &p_hat ).unwrap();

    let x_hat2 = p_hat * x_hat.clone();
    compute_symbol_errors( &x_hat2, &x, None, None ).unwrap() 
}
// end matrix generation functions@}

#[allow(dead_code)]
/// If x is within tol of {-1,0,1}, then return how far off it is
/// otherwise return 0
fn near_pm_one_or_zero( x: f64, tol: f64 ) -> f64 {
    if ( x - 1f64 ).abs() < tol {
        return x - 1f64;
    } else if ( x + 1f64 ).abs() < tol {
        return x + 1f64;
    } else if x.abs() < tol {
        return x;
    }

    0f64
}

#[allow(dead_code)]
/// The main centering step.  Returns a new version of y so that the entries
/// of u*y that are within tol of {-1, 0, 1} are forced to those values
/// Set tol=0f64 and the function will just return a copy of y
fn center_y( u: &na::DMatrix<f64>, y: &na::DMatrix<f64>, tol: f64 ) -> 
    Option<na::DMatrix<f64>> 
{
    let n = y.nrows();
    let k = y.ncols();

    //Skip if tolerance is zero (i.e. don't center)
    if tol == 0f64 {
        return Some( y.clone() );
    }

    let mut uy = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
    u.mul_to( y, &mut uy );

    //Find epsilon
    let del = uy.map( |x| near_pm_one_or_zero(x, tol) );

    //center y, first we need bfs_inv
    match u.clone().try_inverse() {
        Some(u_inv) => return Some( y - u_inv * del ),
        None => return None,
    }
}
