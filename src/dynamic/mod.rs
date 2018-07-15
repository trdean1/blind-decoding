extern crate nalgebra as na;
extern crate rand;

use rand::distributions::{Normal, Distribution};
use std;

use ZTHRESH;

mod feasibleregion;

#[allow(dead_code)]
#[derive(Hash, Eq, PartialEq)]
pub enum BFSType { //{@
    PM1,
    PM10,
    Wrong,
} //@}

// BFS finder functions.{@

//{@
/// Calculate gradient: (x^-1).transpose.
//@}
fn objgrad(x: &mut na::DMatrix<f64>) -> Option<na::DMatrix<f64>> { //{@
    match x.try_inverse_mut() {
        true => Some(x.transpose()),
        false => None,
    }
} //@}

//{@
/// Returns a boolean matrix where each entry tells whether or not the
/// constraint is active.
/// Input:   u = feasible solution (n x n)
///          y = matrix of received symbols (n x k)
/// Output:  Boolean matrix (n x k) where output[i, j] = true iff the ith row of
/// u causes the jth constraint of UY to be active, |<u_i, y_j>| = 1.        
//@}
fn get_active_constraints_bool(prod: &na::DMatrix<f64>,
                               c_bool: &mut na::DMatrix<bool>, zthresh: f64) {
    /*
    let prod = match prod_maybe {
        Some(r) => r,
        None => {
            let r = u * y;
            &r
        },
    };*/

    let (n, k) = prod.shape();

    for j in 0 .. k {
        let mut c_bool_col = c_bool.column_mut(j);
        let prod_col = prod.column(j);
        for i in 0 .. n {
            let entry = prod_col[i];
            c_bool_col[i] =  (1.0 - entry).abs() < zthresh 
                || (1.0 + entry).abs() < zthresh;
        }
    }
}

/*
fn update_active_constraints_array(u: &na::DMatrix<f64>, y: &na::DMatrix<f64>,
                                 out: &Vec<(usize,usize)>, zthresh: f64) {
    let prod = u * y;
    let (n,k) = prod.shape();

    for i in 0 .. n {
        for j in 0 .. k {
            if (1.0 - prod[(i,j)].abs()).abs() < zthresh {
                out.push((i,j));
            }
        }
    }
}
*/

//{@
/// Calculate the distance to the problem boundary along a given vector.
/// Input:   u = current feasible solution
///          v = vector along which to travel
///          y = received symbols
///          mask = points to consider when calculating feasibility; if None,
///          all points are considered.
/// Output: t = maximum distance such that u + t * v remains feasible.
//@}
fn boundary_dist(uy: &na::DMatrix<f64>, dy: &na::DMatrix<f64>,
                 y: &na::DMatrix<f64>, mask: Option<&na::DMatrix<bool>>) -> f64 {
    let mut t_min = std::f64::MAX;

    debug!("uy={}dy={}",uy,dy);
    // Find the lowest value of t such that U + t * V reaches the boundary.
    for j in 0 .. y.shape().1 {
        for i in 0 .. y.shape().0 {
            if let Some(mask) = mask {
                //if mask.column(j)[i] { continue; }
                if mask[(i,j)] { continue; }
            }

            // Determine value of t s.t. [i, j] constr reaches boundary.
            //match dy.column(j)[i].partial_cmp(&0.0) {
            match dy[(i,j)].partial_cmp(&0.0) {
                Some(v) => {
                    let t = match v {
                        std::cmp::Ordering::Less =>
                            //(-1.0 - uy.column(j)[i]) / dy.column(j)[i],
                            (-1.0 - uy[(i,j)]) / dy[(i,j)],
                        std::cmp::Ordering::Greater =>
                            //(1.0 - uy.column(j)[i]) / dy.column(j)[i],
                            (1.0 - uy[(i,j)]) / dy[(i,j)],
                        std::cmp::Ordering::Equal => std::f64::MAX,
                    };
                    if t.abs() < t_min.abs() { t_min = t; }
                },
                None => continue,
            }
        }
    }

    t_min
} //@}
//{@
/// Find an initial BFS for the constraints |UY|\_infty <= 1.  Algorithm works
/// by moving to the problem boundary, then following the projection of the
/// gradient onto the nullspace of active constraints until hittin the boundary
/// again.
/// Input:   u_i = initial estimate for inverse of channel gain matrix
///          y = n x k matrix of received symbols
/// Output:  u = estimated inverse channel gain matrix that forms a BFS
//@}
pub fn find_bfs(u_i: &na::DMatrix<f64>, y: &na::DMatrix<f64>) 
    -> Option<na::DMatrix<f64>> 
{ //{@

    let mut u = u_i.clone();
    let mut gradmtx = u.clone();
    let (n, k) = y.shape();
    let mut v; 
    match objgrad(&mut gradmtx) {
        Some(r) => v = r,
        None => return None,
    }

    let mut uy = u.clone() * y.clone();
    let mut vy = v.clone() * y.clone();
    let mut t = boundary_dist(&uy, &vy, &y, None);
    v *= t;
    u += v;

    let mut gradmtx = na::DMatrix::from_column_slice(n, n, &vec![0.0; n*n]);
    let mut p_bool = na::DMatrix::from_column_slice(n, k, &vec![false; n*k]);
    let mut p_bool_iter = na::DMatrix::from_column_slice(n, k, &vec![false; n*k]);
    let mut p_bool_updates = Vec::with_capacity(k);
    let mut fs = feasibleregion::FeasibleRegion::new(y, None);

    for _iter in 0 .. (n*n - 1) {
        u.mul_to(&y, &mut uy);

        //Print UY to trace each iteration if debug build
        if cfg!(build = "debug") {
            trace!("Iteration {}\nuy = {:.3}\nfs = {:.3}", _iter, uy, fs);
        }

        //TODO: make this more efficient. Not sure the best way yet, but 
        //shouldn't take 2n^2 
        get_active_constraints_bool(&uy, &mut p_bool_iter, ZTHRESH);
        p_bool_updates.clear();
        for j in 0 .. k {
            //let col_iter = p_bool_iter.column(j);
            //let col_orig = p_bool.column_mut(j);
            for i in 0 .. n {
                //if col_iter[i] && !col_orig[i] {
                if p_bool_iter[(i,j)] && !p_bool[(i,j)] {
                    p_bool_updates.push((i, j));
                    p_bool[(i,j)] = true;
                }
            }
        }
        //p_bool.copy_from(&p_bool_iter);

        fs.insert_from_vec( &p_bool_updates );

        gradmtx.copy_from(&u);
        gradmtx = match objgrad(&mut gradmtx) {
            Some(grad) => grad,
            None => break,
        };

        debug!("Iteration {} has {} independent constraints", _iter, fs.get_len_p());

        gradmtx = fs.reject_mtx( &gradmtx );

        debug!("norm = {}", gradmtx.norm());
        if gradmtx.norm() < 1e-9 {
            debug!("Iteration {} gradmtx.norm negligible", _iter);
            break
        }

        gradmtx.mul_to( &y, &mut vy );
        t = boundary_dist(&uy, &vy,
                          &y, Some(&p_bool));
        //gradmtx.apply(|e| e * t);

        u += t*gradmtx.clone();
        /*
        for j in 0 .. n {
            let mut col_u = u.column_mut(j);
            let col_grad = gradmtx.column(j);
            for i in 0 .. n {
                col_u[i] += col_grad[i];
            }
        }
        */
    }

    //Check if we are in {-1, 0, 1} if not, call find_vertex_on_face
    if verify_bfs( &u, &y, ZTHRESH ) == BFSType::Wrong {
        u = match find_vertex_on_face( &u, &y, &fs, ZTHRESH ) {
            Some(r) => r,
            None => u,
        };
    }

    Some(u)
} //@}
//{@
/// Verify that every entry of |uy| == 1.
//@}
#[allow(dead_code)]
pub fn verify_bfs(u: &na::DMatrix<f64>, y: &na::DMatrix<f64>, zthresh: f64) //{@
        -> BFSType {
    if u.determinant().abs() < zthresh * 10e4 {
        return BFSType::Wrong;
    }
    let prod = u * y;
    if prod.iter().all(|&elt| (elt.abs() - 1.0).abs() < zthresh) {
        return BFSType::PM1;
    } else if prod.iter().all(|&elt| (elt.abs() - 1.0).abs() < zthresh
            || elt.abs() < zthresh) {
        return BFSType::PM10;
    }
    BFSType::Wrong
} //@}
// end BFS finder.@}

fn row_to_vertex( u: &na::DMatrix<f64>, y: &na::DMatrix<f64>, 
                  fs: &mut feasibleregion::FeasibleRegion, row: usize,
                  zthresh: f64)
    -> Option<na::DMatrix<f64>>
{
    let (n,_k) = y.shape();
    let mut u_row = na::DMatrix::from_column_slice( 1,n, &vec![0.0; n] );
    u_row.copy_from( &u.row(row) );
    let mut bad_row = u_row.clone() * y.clone();

    //Get a list of entries not in -1, 0, 1
    let mut bad_indices: Vec<usize> = bad_row.into_iter()
             .enumerate()
             .filter(|&(_i,x)| !(x.abs() < zthresh || (x.abs() - 1.0).abs() < zthresh) )
             .map(|(i,_x)| i )
             .collect();

    //Now main loop...don't try moving in more than n subspaces
    let mut i = 0;
    let mut j = 0;
    let mut norm = na::DMatrix::from_column_slice(1, 1, &vec![0.0; 1]);
    while i < n && j < bad_indices.len() {
        trace!("\n Fixing ({}, {})", row, bad_indices[j]);
        //This is the corresponding symbol that lead to the bad <u,y>
        let bad_y = y.column( bad_indices[j] );

        //This is the direction we are going to move in
        let mut v = na::RowDVector::from_column_slice(n, &vec![0.0; n]);
        norm[(0,0)] = 0.0;

        //Attempt to find vector in null space
        let mut k = n;
        while norm[(0,0)] < zthresh {
            //If we didn't succeed n times, then there probably is no nullspace
            k -= 1;
            if k == 0 {
                trace!("Failed to find vector in nullspace!");
                return Some( u_row );
            }

            v = rand_unit( n ).transpose();
            v = fs.reject_vec( &v.transpose(), row ).transpose(); 

            //Update norm of projection
            v.mul_to(&v.transpose(), &mut norm);
            

            if norm[(0,0)] < 1e-13 {
                continue;
            }
            v /= norm[(0,0)].sqrt();
            

            //println!("Normalized direction after rejection={:.4}", v);
        }

        //Compute values to force to -1 and +1
        let mut uv = na::DMatrix::from_column_slice(1,1,&vec![0.0;1]);
        let mut uy_dot = na::DMatrix::from_column_slice(1, 1, &vec![0.0; 1]);
        u_row.mul_to( &bad_y, &mut uy_dot );
        v.mul_to( &bad_y, &mut uv);
        let t_plus = (1.0 - uy_dot[(0,0)]) / uv[(0,0)];
        let t_minus = (-1.0 - uy_dot[(0,0)]) / uv[(0,0)];
        
        //let mut u_row_new = na::DMatrix::from_column_slice(1, n, &vec![0.0; n]);

        let u_plus  = u_row.clone() + t_plus * v.clone();
        let u_minus = u_row.clone() + t_minus * v.clone();

        let uy_plus = u_plus.clone() * y.clone();
        let uy_minus = u_minus.clone() * y.clone();

        let plus_feasible = uy_plus.iter().all(|&elt| elt.abs() < 1.0 + zthresh );
        let minus_feasible = uy_minus.iter().all(|&elt| elt.abs() < 1.0 + zthresh );

        if plus_feasible && minus_feasible {
            //See which direction has more \pm1 entries.

            let plus_pm1 = uy_plus.iter()
                                  .filter(|x| (x.abs() - 1.0).abs() < zthresh )
                                  .fold(0, |acc, _e| acc + 1);

            let minus_pm1 = uy_minus.iter()
                                    .filter(|x| (x.abs() - 1.0).abs() < zthresh )
                                    .fold(0, |acc, _e| acc + 1);

            if minus_pm1 > plus_pm1 {
                u_row.copy_from(&u_plus);
            } else {
                u_row.copy_from(&u_minus);
            }
        } else if plus_feasible {
            u_row.copy_from(&u_plus);
        } else if minus_feasible {
            u_row.copy_from(&u_minus);
        } else {
            //This face is infeasible, try picking a new target
            trace!("This face is infeasible!");
            j += 1;
            continue;
        }

        //We've found a new constraint.  Update and continue
        trace!("New row: {:.4}", u_row.clone() * y.clone());
        bad_row = u_row.clone() * y.clone();

        i += 1;
        j += 1;

        //Check if the row is fixed
        if bad_row.iter()
                  .all(|&elt|( (elt.abs() - 1.0).abs() < zthresh || elt.abs() < zthresh )) {
                    break;
        }

        //If not update p and bad_indices and continue
        bad_indices = bad_row.into_iter()
             .enumerate()
             .filter(|&(_i,x)| !(x.abs() < zthresh || (x.abs() - 1.0).abs() < zthresh) )
             .map(|(i,_x)| i )
             .collect();

        let p_updates = bad_row.into_iter()
             .enumerate()
             .filter(|&(_i,x)| x.abs() < zthresh || (x.abs() - 1.0).abs() < zthresh )
             .map(|(i,_x)| (row,i) )
             .collect();


        fs.insert_from_vec( &p_updates );
    }
     
    Some( u_row )
}

pub fn find_vertex_on_face( u_i : &na::DMatrix<f64>, y: &na::DMatrix<f64>, 
                            d_fs: &feasibleregion::FeasibleRegion, zthresh : f64 )
    -> Option<na::DMatrix<f64>>
{
    let mut u = u_i.clone();
    let uy = u.clone() * y.clone();
    let (n, _k) = y.shape();

    //TODO: probably should just make dynamic a class and then I don't 
    //need to copy this here...
    let mut fs = feasibleregion::FeasibleRegion::from_copy( d_fs );
    let mut updates: Vec<(usize, usize)> = Vec::new();
    for j in 0 .. uy.ncols() {
        for i in 0 .. uy.nrows() {
            if (uy[(i,j)].abs() - 1.0).abs() < zthresh {
                updates.push( (i,j) );
            }
        }
    }
    fs.insert_from_vec( &updates );

    trace!("Starting uy = {:.4}", uy);

    //Find the first row that is not in {-1, 0, 1}
    for i in 0..n {
        let row = uy.row(i);
        if row.iter().all(|&elt| (elt.abs() - 1.0).abs() < zthresh
                          || elt.abs() < zthresh ) {
            continue;
        } else {
            let new_row = match row_to_vertex( &u, &y, &mut fs, i, zthresh ) {
                Some(r) => r,
                None => return None,
            };
            u.row_mut(i).copy_from(&new_row);

            //If above call put matrix into {-1,0,1} then bail
            if verify_bfs( &u, &y, zthresh ) != BFSType::Wrong {
                break;
            }
        }
    }

    trace!("Ending uy = {:.4}", u.clone() * y.clone() );

    Some( u )
}

#[allow(dead_code)]
pub fn rand_unit(n: usize) -> na::DVector<f64> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n);
    let dist = Normal::new(0.0, 1.0);
    for _ in 0 .. (n) {
        data.push(dist.sample(&mut rng));
    }

    let mut v = na::DVector::from_column_slice(n, &data);
    let norm = v.norm();
    v /= norm;

    return v;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This tests the BFS Finder with a fixed input.  The BFS solver will return
    /// something infeasible half the time. This works fine in the python version
    #[test]
    fn bfs_instability_test() {
        for _ in 0 .. 500 {
            let y = na::DMatrix::from_row_slice( 4,6,
                    &vec![
                    -0.85105277, -0.63492999, 3.41237287, 1.20811145, 0.35892148, 2.41836825,
                    3.96976924, -0.73519243, -1.04010774, 0.12971196, 1.03251896, -2.80775547,
                    3.12355480, 2.95773796, 0.67356399, 0.87056107, 2.94202708, 0.68919778,
                    -2.31709338, 0.66587824, 1.21174544, 2.38172050, -1.41046808, 3.28820386
                    ]);

            let ui = na::DMatrix::from_row_slice( 4,4,
                   &vec![
                    -0.06779237, -0.08776801,  0.03347982, -0.04695821,
                    -0.01830087, -0.05418955, -0.09233134,  0.06187484,
                    0.06205623, -0.02276697, -0.05832655, -0.08862112,
                    0.08272422, -0.06683513,  0.05076455,  0.04168607
                   ]);

            let bfs = find_bfs(&ui, &y).unwrap();
            
            let uy = bfs * y;
            let feasible = uy.iter().all( |&elt| elt < 1.0 + 1e-6 );
            
            if !feasible{ println!("UY = {:.4}", uy); }

            assert!( feasible );
        }
    }

    /*
    /// This was in the old code...not sure what bug this was testing
    fn use_given_matrices() -> (na::DMatrix<f64>, na::DMatrix<f64>) { 
        let y = na::DMatrix::from_row_slice(3, 8,
                &vec![
                -2.13221971, 1.65267722, -3.58171708, 0.20317985,
                -0.20317985, -1.65267722, -2.13221971, 2.13221971,

                1.33427445, 0.75515798, -1.76382965, -2.34294612, 
                2.34294612, -0.75515798, 1.33427445, -1.33427445,

                -2.49940914, 0.07649133, -0.74610697, 1.8297935,
                -1.8297935, -0.07649133, -2.49940914, 2.49940914,
                ]);

        let u_i = na::DMatrix::from_row_slice(3, 3,
                &vec![
                -0.04251098, -0.1185273, 0.21597244,
                0.22137548, 0.07778901, 0.08626569,
                0.10810049, -0.20591296, -0.0917286,
                ]);
        
        (y, u_i)
    }
    */
}
