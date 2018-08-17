extern crate nalgebra as na;
extern crate rand;

use std;
use std::fmt;

use super::matrix;

mod feasibleregion;

#[allow(dead_code)]
#[derive(Hash, Eq, PartialEq)]
pub enum BFSType { //{@
    PM1,
    PM10,
    Wrong,
} //@}

impl fmt::Display for BFSType { //{@
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &BFSType::PM1 => write!(f, "{}", "PM1"),
            &BFSType::PM10 => write!(f, "{}", "PM10"),
            &BFSType::Wrong => write!(f, "{}", "Wrong"),
        }
    }
} //@}

pub enum BfsError {
    SingularUi,
    SingularU,
    RowToVertFailure,
}

pub struct BfsFinder {
    y: na::DMatrix<f64>,
    active_constraints: Vec<Vec<bool>>,
    fs: feasibleregion::FeasibleRegion,

    u: na::DMatrix<f64>,
    v: na::DMatrix<f64>,
    grad: Option<na::DMatrix<f64>>,
    uy: na::DMatrix<f64>,
    vy: na::DMatrix<f64>,

    n: usize,
    k: usize,
    zthresh: f64,
}

impl Default for BfsFinder {
    fn default() -> BfsFinder {
        BfsFinder {
            y: na::DMatrix::from_column_slice( 0, 0, &Vec::new() ),
            active_constraints: Vec::new(),
            fs: feasibleregion::FeasibleRegion::default(),

            u: na::DMatrix::from_column_slice( 0, 0, &Vec::new() ),
            v: na::DMatrix::from_column_slice( 0, 0, &Vec::new() ),
            grad: None,
            uy: na::DMatrix::from_column_slice( 0, 0, &Vec::new() ),
            vy: na::DMatrix::from_column_slice( 0, 0, &Vec::new() ),

            n: 0,
            k: 0,
            zthresh: 0.0,
        }
    }
}

impl BfsFinder {
    pub fn new( y: &na::DMatrix<f64>, zthresh: f64 ) 
        -> BfsFinder {
        let (n, k) = y.shape();
        BfsFinder {
            y: y.clone(),
            active_constraints: vec![ vec![false; k]; n ],
            fs: feasibleregion::FeasibleRegion::new( y, Some(zthresh) ),

            u: na::DMatrix::zeros( n, n ),
            v: na::DMatrix::zeros( n, n ),
            grad: None,
            uy: na::DMatrix::zeros( n, k ),
            vy: na::DMatrix::zeros( n, k ),

            n: n,
            k: k,
            zthresh: zthresh,
        }
    }

    fn update_grad( &mut self ) {
        self.grad = match self.u.clone().try_inverse() {
            Some(gr) => Some(gr.transpose()),
            None => None
        }
    }

    /// If update_fs is set to true then this will add newly activated
    /// constraints to the data structure FeasibleRegion
    fn update_active_constraints( &mut self, update_fs: bool ) {
        for i in 0 .. self.n {
            for j in 0 .. self.k {
                let entry = self.uy[(i,j)];
                let old = self.active_constraints[i][j];
                self.active_constraints[i][j] = (1.0 - entry).abs() < self.zthresh 
                    || (1.0 + entry).abs() < self.zthresh;
                if update_fs && 
                   old == false &&
                   self.active_constraints[i][j] == true {
                    self.fs.insert( i, j );
                }
            }
        }
    }

    //{@
    /// Calculate the distance to the problem boundary along a given vector.
    /// Input:   u = current feasible solution
    ///          mask = points to consider when calculating feasibility; if None,
    ///          all points are considered.
    /// Output: t = maximum distance such that u + t * v remains feasible.
    //@}
    fn boundary_dist( &self, mask: Option<&Vec<Vec<bool>>> )
        -> f64 {
        let mut t_min = std::f64::MAX;

        for j in 0 .. self.k {
            for i in 0 .. self.n {
                if let Some(mask) = mask {
                    if mask[i][j] { continue; }
                }

            // Determine value of t s.t. [i, j] constr reaches boundary.
            match self.vy[(i,j)].partial_cmp(&0.0) {
                Some(v) => {
                    let t = match v {
                        std::cmp::Ordering::Less =>
                            (-1.0 - self.uy[(i,j)]) / self.vy[(i,j)],
                        std::cmp::Ordering::Greater =>
                            (1.0 - self.uy[(i,j)]) / self.vy[(i,j)],
                        std::cmp::Ordering::Equal => std::f64::MAX,
                    };
                    if t.abs() < t_min.abs() { t_min = t; }
                },
                None => continue,
            }
            }
        }

        t_min
    }

    /// 
    /// Find an initial BFS for the constraints |UY|\_infty <= 1.  Algorithm works
    /// by moving to the problem boundary, then following the projection of the
    /// gradient onto the nullspace of active constraints until hittin the boundary
    /// again.
    /// Input:   u_i = initial estimate for inverse of channel gain matrix
    /// Output u = estimated inverse channel gain matrix or an error code
    ///
    pub fn find_bfs( &mut self, u_i: &na::DMatrix<f64> ) -> Result<na::DMatrix<f64>,BfsError> {
        assert!( self.n == u_i.shape().0 );

        self.u.copy_from(u_i);
        self.update_grad();
        if self.grad.is_none() {
            return Err(BfsError::SingularUi);
        }

        //First step...move along gradient until we hit the boundary
        self.u.mul_to( &self.y, &mut self.uy );
        self.v.copy_from( self.grad.as_ref().unwrap() );
        self.v.mul_to( &self.y, &mut self.vy );
        let mut t = self.boundary_dist( None );
        self.v *= t;
        self.u += self.v.clone();

        //Update UY
        self.u.mul_to( &self.y, &mut self.uy );

        for _iter in 0 .. (self.n*self.n - 1) {
            //Update active constraints and feasible region
            self.update_active_constraints( true );

            //Update gradient and find v by rejecting grad from active constraints
            self.update_grad();
            if self.grad.is_none() {
                return Err(BfsError::SingularU);
            }

            self.v = self.fs.reject_mtx( self.grad.as_ref().unwrap() );

            //If true, then the active constraints are full rank and we have nowhere to go
            if self.v.norm() < 1e-12 {
                break
            }

            //Now move in the direction of v until we no longer can
            self.v.mul_to( &self.y, &mut self.vy );
            t = self.boundary_dist( Some(&self.active_constraints) );

            //Update u and uy
            self.u += t * self.v.clone();
            self.u.mul_to( &self.y, &mut self.uy );
        }

        if self.verify_bfs() == BFSType::Wrong {
            match self.find_vertex_on_face() {
                Ok(_) => return Ok(self.u.clone()),
                Err(e) => return Err(e),
            };
        }

        Ok(self.u.clone())
    }
 
    /// 
    /// Returns the type of BFS currently stored in uy,
    /// i.e. |UY| == 1, or {-1, 0, 1} or other
    ///
    pub fn verify_bfs( &self ) -> BFSType {
        //Not sure if this is needed?
        if self.u.determinant().abs() < self.zthresh * 1e5 {
            return BFSType::Wrong;
        }

        let mut found_zero = false;
        for &elt in self.uy.iter() {
            //Check if {-1, +1}
            if (elt.abs() - 1.0).abs() > self.zthresh {
                if elt.abs() < self.zthresh {
                    found_zero = true;
                } else {
                    //If any entry is not in {-1,0,1} then it's wrong
                    return BFSType::Wrong;
                }
            }
        }

        if found_zero {
            return BFSType::PM10;
        } else {
            return BFSType::PM1;
        }
    }

    /// If a solution has entries that are not in {-1, 0, 1}, then we will 
    /// attempt to fix them by going one row at a time, moving in the nullspace
    /// of the active constraints.
    fn find_vertex_on_face( &mut self ) -> Result<(), BfsError> {
        //Find the first row that is not in {-1, 0, 1}
        for i in 0..self.n {
            if self.uy.row(i).iter().all(|&elt| (elt.abs() - 1.0).abs() < self.zthresh
                              || elt.abs() < self.zthresh ) {
                continue;
            } else {
                //Try to move in the nullspace to push to {-1, 0, 1}
                if let Err(_) = self.row_to_vertex( i ) {
                    return Err(BfsError::RowToVertFailure);
                }

                //If above call put matrix into {-1,0,1} then we are done 
                if self.verify_bfs() != BFSType::Wrong {
                    break;
                }
            }
        }

        Ok(()) 
    }

    ///
    /// Subroutine of find_vertex_on_face.  Attempts to move row to make all entries
    /// \pm 1.  Greedily picks the best direction (most \pm 1 values)
    ///
    fn row_to_vertex( &mut self, row: usize ) -> Result<(), BfsError> 
    {
        let zthresh = self.zthresh;

        let mut u_row = self.u.row_mut( row );

        //Get a list of entries not in -1, 0, 1
        let mut bad_indices: Vec<usize> = self.uy.row( row ).into_iter()
                 .enumerate()
                 .filter(|&(_i,x)| !(x.abs() < zthresh || (x.abs() - 1.0).abs() < zthresh) )
                 .map(|(i,_x)| i )
                 .collect();

        //Now main loop...don't try moving in more than n subspaces
        let mut i = 0;
        let mut j = 0;
        let mut norm = na::DMatrix::from_column_slice(1, 1, &vec![0.0; 1]);
        while i < self.n && j < bad_indices.len() {
            //This is the corresponding symbol that lead to the bad <u,y>
            let bad_y = self.y.column( bad_indices[j] );

            //This is the direction we are going to move in
            let mut v = na::RowDVector::zeros(self.n);
            norm[(0,0)] = 0.0;

            //Attempt to find vector in null space
            let mut k = self.n;
            while norm[(0,0)] < self.zthresh {
                //If we didn't succeed n times, then there probably is no nullspace
                k -= 1;
                if k == 0 {
                    return Ok(());
                }

                v = matrix::rand_unit( self.n ).transpose();
                v = self.fs.reject_vec( &v.transpose(), row ).transpose(); 

                //Update norm of projection
                v.mul_to(&v.transpose(), &mut norm);
                

                if norm[(0,0)] < 1e-13 {
                    continue;
                }
                v /= norm[(0,0)].sqrt();
            }

            //Compute values to force to -1 and +1
            let mut uv = na::DMatrix::from_column_slice(1,1,&vec![0.0;1]);
            let mut uy_dot = na::DMatrix::from_column_slice(1, 1, &vec![0.0; 1]);
            u_row.mul_to( &bad_y, &mut uy_dot );
            v.mul_to( &bad_y, &mut uv);
            let t_plus = (1.0 - uy_dot[(0,0)]) / uv[(0,0)];
            let t_minus = (-1.0 - uy_dot[(0,0)]) / uv[(0,0)];
            
            //Check whether each direction yields something feasible
            let u_plus  = u_row.clone_owned() + t_plus * v.clone();
            let u_minus = u_row.clone_owned() + t_minus * v.clone();

            let uy_plus = u_plus.clone_owned() * self.y.clone();
            let uy_minus = u_minus.clone_owned() * self.y.clone();

            let plus_feasible = uy_plus.iter().all(|&elt| elt.abs() < 1.0 + zthresh );
            let minus_feasible = uy_minus.iter().all(|&elt| elt.abs() < 1.0 + zthresh );

            let new_uy;

            if plus_feasible && minus_feasible {
                //If both directions are feasible, be greedy and pick the direction with more
                //good entries
                let plus_pm1 = uy_plus.iter()
                                      .filter(|x| (x.abs() - 1.0).abs() < zthresh )
                                      .fold(0, |acc, _e| acc + 1);

                let minus_pm1 = uy_minus.iter()
                                        .filter(|x| (x.abs() - 1.0).abs() < zthresh )
                                        .fold(0, |acc, _e| acc + 1);

                if minus_pm1 > plus_pm1 {
                    u_row.copy_from(&u_plus);
                    new_uy = uy_plus;
                } else {
                    u_row.copy_from(&u_minus);
                    new_uy = uy_minus;
                }
            } else if plus_feasible {
                u_row.copy_from(&u_plus);
                new_uy = uy_plus;
            } else if minus_feasible {
                u_row.copy_from(&u_minus);
                new_uy = uy_minus;
            } else {
                //This face is infeasible, try picking a new target
                j += 1;
                continue;
            }

            //We've found a new constraint.  Update and continue
            for jj in 0 .. self.k {
                self.uy[(row, jj)] = new_uy[(0,jj)];
            }

            i += 1;
            j += 1;

            //Check if the row is fixed, if not update p and bad_indices and continue
            //Note we could call update_constraints() but that will check all n*k entries
            //versus just the k in the row we are fiddling with
            bad_indices.clear();
            for (i, x) in new_uy.into_iter().enumerate() {
                if !(x.abs() < zthresh || (x.abs() - 1.0).abs() < zthresh) {
                    bad_indices.push( i );
                } else {
                    if self.active_constraints[row][i] == false {
                        self.fs.insert(row, i);
                        self.active_constraints[row][i] = true;
                    }
                }
            }
            
            if bad_indices.len() == 0 {
                break;
            }
        }
         
        Ok(()) 
    }
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

            //let bfs = find_bfs(&ui, &y).unwrap();
            let mut bfs_finder = BfsFinder::new( &y, 1e-9 );
            if let Ok(bfs) = bfs_finder.find_bfs( &ui ) {
                let uy = bfs * y;
                let feasible = uy.iter().all( |&elt| elt < 1.0 + 1e-6 );
            
                if !feasible{ println!("UY = {:.4}", uy); }

                assert!( feasible );
            }
        }
    }

    #[test]
    fn new_solver_basic_test() {
        for _ in 0 .. 100 {
            let dims = vec![(4,8)];

            let x = matrix::get_matrix(&dims[0 .. 1]);
            let (_a, y) = matrix::y_a_from_x(&x, false );
            let u_i = matrix::rand_init(&y);

            let mut bfs_finder = BfsFinder::new( &y, 1e-9 );
            let u = bfs_finder.find_bfs( &u_i );

            if let Err(_) = u { 
                assert!(false, "find_bfs returned error");
            }
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
