/*
 * This code is proprietary information and property of Stanford University.
 * Do not share without permission of the authors.  This code may only be
 * used for educational purposes.  
 *
 * Authors: Jonathan Perlstein (jrperl@cs.stanford.edu)
 *          Thomas Dean (trdean@stanford.edu)
 *
 */
#![feature(test)]

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
pub mod sparsemtx;

use tableau::FlexTabError;
use tableau::FlexTab;
use testlib::TrialResults;

const ZTHRESH: f64 = 1e-9;

pub struct Solver {
    stats:          TrialResults,
    max_attempts:   usize,
    center_tol:     f64,
    timer:          std::time::Instant,
    n_tx:           usize,
    m_rx:           usize,
}

impl Default for Solver {
    fn default() -> Solver {
        Solver {
            stats:          TrialResults::new(0,0,0,0.0),
            max_attempts:   100,
            center_tol:     0.0,
            timer:          std::time::Instant::now(),
            n_tx:           0,
            m_rx:           0,
        }
    }
}

impl Solver {
    pub fn new( n_tx: usize, m_rx: usize, center_tol: f64, attempts: usize )
            -> Solver {
        assert!(m_rx >= n_tx, format!("error: m_rx = {}, n_tx = {}", m_rx, n_tx));
        Solver {
            n_tx,
            m_rx,
            max_attempts: attempts,
            center_tol: center_tol,
            ..Default::default()
        }
    }

    /// Perform a single run using a given +y+ matrix, which contains k symbols each of length n.
    pub fn solve(&mut self, y: &na::DMatrix<f64>) -> Result<FlexTab, FlexTabError>
    {
        assert!(y.nrows() == self.m_rx, format!("y.shape = {:?}, m_rx = {}", y.shape(), self.m_rx));
        self.timer = ::std::time::Instant::now();
        let svd = na::SVD::new(y.clone(), true, false);
        let s = svd.singular_values;
        let r = s.iter().filter(|&elt| *elt > ZTHRESH).count();
        if r < self.n_tx {
             self.stats.time_elapsed += self.time_elapsed();
             return Err(FlexTabError::SingularInput);
        }

        let mut y = y.clone();
        if self.m_rx > self.n_tx {
            let u = match svd.u {
                Some(u) => u,
                None => return Err(FlexTabError::SingularInput),
            };
            let u = u.remove_rows(self.n_tx, self.m_rx - self.n_tx);
            y = u * y
        }
        self.stats.trials += 1;

        let mut attempts = 0;
        let mut ft;
        let mut best: Option<FlexTab> = None;

        //If true, check that y is not singular.
        //Solver should still run if check fails but may be slow and will likely 
        //return garbage

        let mut bfs_finder = dynamic::BfsFinder::new( &y, ZTHRESH );
        let mut z; //Holds centered version of y
        // Loop trying new u_i until we get n linearly independent \pm 1 cols.
        loop {
            attempts += 1;
            if attempts > self.max_attempts {
                self.stats.time_elapsed += self.time_elapsed();

                match best {
                    Some(b) => { 
                        return Ok(b);
                    },
                    None => {
                        self.stats.runout += 1;
                        return Err(FlexTabError::Runout)
                    }
                };
            }
            let mut u_i = matrix::rand_init(&y); // Choose rand init start pt.
            //let (y, u_i) = use_given_matrices(); // Use for debugging only.

            z = y.clone();

            //Centering loop
            let mut bfs;
            let mut center_attempts = 0;
            let mut bfs_fail = false;
            //let mut z_last;
            loop {
                debug!("Center attempt: {}", center_attempts);
                //match dynamic::find_bfs(&u_i, &z) {
                match bfs_finder.find_bfs(&u_i) {
                    Ok(b) => bfs = b,
                    Err(_) => {
                        //This shouldn't really ever happen but does if the input
                        //is poorly conditioned and we run into numerical stability 
                        //issues
                        bfs_fail = true;
                        bfs = u_i;
                        self.stats.badstart += 1;
                        break;
                    },
                }

                if self.center_tol == 0f64 { break; }

                if (bfs.clone() - u_i.clone()).amax() < ZTHRESH {
                    break;
                } else {
                    u_i = bfs.clone();
                }
                //z_last = z.clone();

                //Center y.  If this fails (it never should) then just use
                //the old value of y and try our luck with FlexTab
                z = match center_y( &bfs, &z, self.center_tol ) {
                    Some(yy) => yy,
                    None => z,
                };

                //if (z.clone() - z_last.clone()).amax() < ZTHRESH {
                //    break;
                //}

                center_attempts += 1;
                self.stats.centerattempts += 1;
                if center_attempts >= z.shape().0 { break; }

                //XXX Need to write interface to update this
                bfs_finder = dynamic::BfsFinder::new( &z, ZTHRESH );
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
                        self.stats.goodcols += 1;
                        continue;
                    },
                    // Any other error => propogate up.
                    _ => {
                        self.stats.error += 1;
                        self.stats.time_elapsed += self.time_elapsed();
                        return Err(e);
                    },
                },
            };

            // Now we have at least n good cols, so try to solve.  If error is that
            // we don't have n linearly independent good cols, then try new u_i.
            // Do same thing if we appear to have been trapped.

            //let starting_bfs = ft.state.get_u();

            match ft.solve() {
                Ok(_) => break,
                Err(e) => match e {
                    FlexTabError::LinIndep 
                        => { 
                            self.stats.linindep += 1;
                            self.stats.numhops += ft.visited_vertices();
                        },
                    FlexTabError::StateStackExhausted => {
                            self.stats.numhops += ft.visited_vertices();
                            best = match best {
                                Some(b) => if ft.state.obj() > b.state.obj()
                                    { Some(ft) } else { Some(b) },
                                None => Some(ft),
                            };
                            self.stats.statestack += 1;

                        },
                    FlexTabError::TooManyHops => {
                            self.stats.numhops += ft.visited_vertices();
                            best = match best {
                                Some(b) => if ft.state.obj() > b.state.obj()
                                    { Some(ft) } else { Some(b) },
                                None => Some(ft),
                            };
                            self.stats.toomanyhops += 1;
                    }
                    FlexTabError::Trapped => {
                            self.stats.numhops += ft.visited_vertices();
                            best = match best {
                                Some(b) => if ft.state.obj() > b.state.obj()
                                    { Some(ft) } else { Some(b) },
                                None => Some(ft),
                            };
                            self.stats.trap += 1;
                    }

                    _ => {
                        self.stats.error += 1;
                        self.stats.time_elapsed += self.time_elapsed();
                        self.stats.numhops += ft.visited_vertices();
                        return Err(e);
                    },
                },
            }
        }

        // If FlexTab is reduced, we need to do this again starting with a real BFS.
        // Here there is no possibility of insufficient good cols or lin indep cols.
        let return_val = Ok(ft);

        /*
        if ft.has_ybad() {
            self.stats.reduced += 1;
            let ftfull = FlexTab::new(&ft.state.get_u(), &z, ZTHRESH)?;
            return_val = Ok(ftfull);
            /*
            return_val = match ftfull.solve() {
                Ok(_) => {
                    //after = Some( ftfull.state.get_u() * z.clone() );
                    Ok(ftfull)
                }
                Err(e) => {
                    warn!("ftfull from reduced err = {}", e);
                    match e {
                        FlexTabError::StateStackExhausted
                            | FlexTabError::TooManyHops
                            => Ok(ftfull),
                        _ => Err(e),
                    }
                }
            };
            */
        } else {
            return_val = Ok(ft);
        }
        */

        self.stats.time_elapsed += self.time_elapsed();
        match return_val {
            Ok(_) => self.stats.complete += 1,
            Err(_) => self.stats.error += 1,
        }

        return_val
    }

    pub fn solve_reps(&mut self, x: &na::DMatrix<f64>, num_reps: u64, complex: bool) {
        for _iter in 0 .. num_reps {
            // Obtain A, Y matrices, then run.
            let (_a, y) = matrix::y_a_from_x(&x, self.m_rx, complex);
            let y_reduced = match matrix::rank_reduce(&y, self.n_tx) {
                Some(y) => y,
                None => { self.stats.error += 1; continue; }
            };
            
            debug!("Y = {:.02}", y);

            match self.solve(&y) {
                Err(_) => {
                    /*match e {
                        FlexTabError::Runout => {
                            res.runout += 1; // ran out
                            debug!("ran out of attempts");
                        },
                        _ => {
                            res.error += 1; // something else -- problem
                            println!("critical error = {}", e);
                        },
                    };*/
                },
                Ok(ft) => {
                    // Obtained a result: check if UY = X up to an ATM.
                    let u = ft.state.get_u();
                    let uy = u * y_reduced.clone();
                    if equal_atm(&uy, &x) {
                        self.stats.success += 1;
                    } else {
                        // UY did +not+ match X, print some results and also
                        // determine if UY was even a vertex.
                        //match a.try_inverse() {
                        //    None => debug!("UNEQUAL: Cannot take a^-1"),
                        //    Some(inv) => debug!("UNEQUAL: u = {:.3}a^-1 = {:.3}", 
                        //            ft.best_state.get_u(), inv),
                        //};
                        if is_pm1(&uy, ft.get_zthresh()) {
                            self.stats.not_atm += 1; // UY = \pm 1
                        } else {
                            self.stats.error += 1; // UY != \pm 1 -- problem
                            debug!("critical error: uy = {:.3}", uy);
                        }
                    }
                },
            };
        }
    }

    pub fn clear_stats( &mut self ) {
        self.stats.clear();
    }

    pub fn get_stats( &self ) -> TrialResults {
        self.stats.clone()
    }

    fn time_elapsed( &self ) -> f64 {
        let elapsed = self.timer.elapsed();
        
        elapsed.as_secs() as f64 + 
        elapsed.subsec_nanos() as f64 * 1e-9
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

pub fn is_pm1( a: &na::DMatrix<f64>, zthresh: f64 ) -> bool {
    a.iter().all(|&e| (e.abs() - 1.0).abs() < zthresh)
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
    //let mut p_hat = na::DMatrix::from_column_slice(n, n, &vec![0.0; n*n]);
    let mut p_hat = na::DMatrix::zeros(n, n);
    u.mul_to(&h, &mut p_hat);
    p_hat.apply( |e| e.round() );
    
    //Check that p_hat is an ATM
    if is_atm( &p_hat ) == false {
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


    //let mut x_tilde = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
    let mut x_tilde = na::DMatrix::zeros(n, k);
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

    //let mut uy = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
    let mut uy = na::DMatrix::zeros(n, k); 
    u.mul_to( y, &mut uy );

    //Find epsilon
    let del = uy.map( |x| near_pm_one_or_zero(x, tol) );

    //center y, first we need bfs_inv
    match u.clone().try_inverse() {
        Some(u_inv) => return Some( y - u_inv * del ),
        None => return None,
    }
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


