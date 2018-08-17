extern crate nalgebra as na;

use std::fmt;

use ZTHRESH;

pub struct FeasibleRegion {
    y: na::DMatrix<f64>,
    dims: (usize,usize),
    //b: Vec<Vec<na::RowDVector<f64>>>,
    p: Vec<Vec<na::RowDVector<f64>>>,
    col_map: Vec<Vec<usize>>,
    zthresh: f64,
}

impl Default for FeasibleRegion {
    fn default() -> FeasibleRegion {
        FeasibleRegion {
            y: na::DMatrix::from_column_slice(0,0, &Vec::new()),
            dims: (0,0),
            //b: vec![vec![na::RowDVector::from_column_slice(0, &Vec::new()); 1]],
            p: vec![vec![na::RowDVector::from_column_slice(0, &Vec::new()); 1]],
            col_map: vec![Vec::new(); 1],
            zthresh: ZTHRESH,
        }
    }
}

impl fmt::Display for FeasibleRegion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();
        s += &format!("Y = {:.4}\n", self.y);

        s += "p = \n";
        for i in 0 .. self.p.len() {
            for j in 0 .. self.p[i].len() {
                s += "\t[";
                for k in 0 .. self.dims.0 {
                    if self.p[i][j][(0,k)] >= 0.0 {
                        s += " ";
                    }
                    s += &format!("{:.4}", self.p[i][j][(0,k)]);
                    if k != self.dims.0 - 1 {
                        s += ", ";
                    }
                }
                s += "]\n";
            }
        }

        s += "Column Mapping = \n";
        for i in 0 .. self.col_map.len() {
            s += &format!("{:?}\n", self.col_map[i]);
        }

        write!(f, "{}", s)
    }
}

impl FeasibleRegion {
    pub fn new( y: &na::DMatrix<f64>, zthresh: Option<f64>) -> FeasibleRegion {
        let (n,k) = y.shape();
        FeasibleRegion {
            y: y.clone(),
            dims: (n,k),
            //b: vec![Vec::with_capacity(k); n],
            p: vec![Vec::with_capacity(n); n],
            col_map: vec![Vec::with_capacity(n); n],
            zthresh: match zthresh{ Some(z) => z, None => ZTHRESH },

            ..Default::default()
        }
    }

    #[allow(dead_code)]
    pub fn from_copy( fs: &FeasibleRegion ) -> FeasibleRegion {
        let new_y = fs.get_y();
        let (n,k) = new_y.shape();

        FeasibleRegion{ 
            y: new_y,
            dims: (n,k),
            p: fs.get_p(),
            col_map: fs.get_col_map().clone(),
            zthresh: fs.get_zthresh(),

            ..Default::default()
        }
    }

    #[allow(dead_code)]
    pub fn insert_from_vec( &mut self, update: &Vec<(usize,usize)> ) {
        for (i,j) in update.iter() {
            self.insert( *i, *j );
        }
    }

    pub fn insert( &mut self, row: usize, column: usize ) {
        //Check to see if we have already added this entry
        if self.col_map[row].iter().any( |&x| x == column ) {
            return;
        }

        //Copy column of y and insert into b
        //self.b[row].push( self.y.column(column).transpose().into_owned() );

        //Update column map
        self.col_map[row].push(column);

        //Update p
        let mut v: na::RowDVector<f64> = 
            self.y.column(column).transpose().into_owned();

        //Normalize v
        let v_norm = v.iter()
                      .fold(0.0,
                            |sum, &e|
                            sum + e * e)
                      .sqrt();

        v /= v_norm;

        //Perform matrix rejection, add v to p as long as it is not orthogonal
        for i in 0 .. self.p[row].len() {
            let u = &self.p[row][i];

            //uv = (u * v.T)
            let mut uv = 0.0;
            for j in 0 .. u.len() {
                uv += u[j] * v[j];
            }

            //u_norm = u * u.T
            let mut uu = 0.0;
            for j in 0 .. u.len() {
                uu += u[j] * u[j];
            }

            //v = v - ((u * v.T) / (u * u.T)) * u
            for j in 0 .. v.len() {
                v[j] -= (uv / uu) * u[j];
            }

            let mut v_norm = 0.0;
            for j in 0 .. v.len() {
                v_norm += v[j] * v[j];
            }
            v_norm.sqrt();

            if v_norm < self.zthresh { return }

            //Normalize and continue
            //v /= v_norm;
            v /= v.norm();
        }
        
        //If p already has n entries then we should have returned already
        if self.p[row].len() >= self.dims.0 {
            for pp in self.p[row].iter() {
                debug!("{:.04}norm: {:.04}", pp, pp.norm());
            }
            assert!(false, "P already has n entries");
        }
        self.p[row].push( v );
    }

    //TODO: only 60% of this is spent in reject_vec_to...the rest is allocation or copying 
    pub fn reject_mtx( &self, v: &na::DMatrix<f64> )
            -> na::DMatrix<f64> {
        let (m,n) = v.shape();
        let mut vv: na::DMatrix<f64> = na::DMatrix::zeros(m,n);
        let mut tmp: na::RowDVector<f64> = na::RowDVector::zeros(n);

        for i in 0 .. m {
            let v_row = v.row( i );
            self.reject_vec_to( &v_row.into_owned(), &mut tmp, i ); 
            for j in 0 .. n {
                vv[(i,j)] = tmp[(0,j)];
            }
        }

        vv
    }

    pub fn reject_vec( &self, v: &na::DVector<f64>, row: usize ) 
            -> na::DVector<f64> {
        let mut vv = v.transpose().into_owned();

        //let mut uu = na::DMatrix::from_column_slice(1,1,&vec![0.0;1]);
        //let mut uv = na::DMatrix::from_column_slice(1,1,&vec![0.0;1]);

        for i in 0 .. self.p[row].len() {
            let u = &self.p[row][i];

            //vv.mul_to( &u.transpose(), &mut uv ); 
            //u.mul_to( &u.transpose(), &mut uu );
            
            //uv = vv.T * u
            let uv = u.iter()
                      .enumerate()
                      .fold(0.0,
                            |sum, (idx, &e)|
                            sum + e * vv[idx]);
            
            //uu = u.T * u
            let uu = u.iter()
                  .enumerate()
                  .fold(0.0,
                        |sum, (idx, &e)|
                        sum + e * u[idx]);

            assert!(uu > 1e-12);

            vv -= (uv / uu) * u;
        }
        
        vv.transpose()
    }

    pub fn reject_vec_to( &self, v: &na::RowDVector<f64>, res: &mut na::RowDVector<f64>, 
                          row: usize ) {
        assert!( v.nrows() == res.nrows() );

        res.copy_from(v);

        for i in 0 .. self.p[row].len() {
            let u = &self.p[row][i];
            
            //uv = vv.T * u
            let mut uv = 0.0;
            for j in 0 .. u.len() {
                uv += u[j] * res[j];
            }
            
            //uu = u.T * u
            let mut uu = 0.0;
            for j in 0 .. u.len() {
                uu += u[j] * u[j];
            }

            assert!(uu > 1e-12);

            //*res -= (uv / uu) * u;
            for j in 0 .. res.len() {
                res[j] -= (uv / uu) * u[j];
            }
        }
    }

    #[allow(dead_code)]
    pub fn get_len_p( &self ) -> usize {
        let mut len = 0;
        for row in self.p.iter() {
            len += row.len();
        }

        len
    }

    #[allow(dead_code)]
    pub fn get_p( &self ) -> Vec<Vec<na::RowDVector<f64>>> {
        self.p.clone()
    }

    #[allow(dead_code)]
    pub fn get_y( &self ) -> na::DMatrix<f64> {
        self.y.clone()
    }

    #[allow(dead_code)]
    pub fn get_col_map( &self ) -> &Vec<Vec<usize>> {
        &self.col_map
    }

    #[allow(dead_code)]
    pub fn get_zthresh( &self ) -> f64 {
        self.zthresh
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let y = na::DMatrix::from_row_slice(4, 9, &vec![
            -2.9559865737125,  2.5865278325864, -2.2980562996102,  0.7369790120865,
            -1.7022470264114,  2.5865278325864,  2.5865278325864, -0.7369790120865,
            -1.9907185593875,
             1.2508223406517, -0.7159569731051,  2.9493672391056, -3.8027781425851,
             3.6435053746187, -0.7159569731051, -0.7159569731051,  3.8027781425851,
             1.4100951086181,
            -0.9635641883542, -1.4738581147596, -2.9837682836541, -1.3922099258609,
            -2.0144903304535, -1.4738581147596, -1.4738581147596,  1.3922099258609,
             2.4431360679602,
            -0.8759863225138, -0.8875888657254,  2.6402078223407, -1.6158556098730,
             1.2462441219873, -0.8875888657254, -0.8875888657254,  1.6158556098730,
            -0.5063748346281]);

        let mut fs = FeasibleRegion::new( &y, Some(ZTHRESH) );
        
        println!("Add first entry");
        fs.insert( 0, 0 );
        println!("{}", fs);

        println!("Make full rank");
        fs.insert( 0, 1 );
        fs.insert( 0, 2 );
        fs.insert( 0, 3 );
        println!("{}", fs);

        println!("Adding redundant constraints");
        fs.insert( 0, 4 );
        fs.insert( 0, 5 );
        fs.insert( 0, 6 );
        fs.insert( 0, 7 );
        println!("{}", fs);

        println!("Ensuring p is orthonormal");
        let pp = fs.get_p();
        let mut res: Vec<f64> = Vec::new();
        for i in 0 .. pp[0].len() {
            for j in 0 .. pp[0].len() {
                let mut prod = pp[0][i].clone() * pp[0][j].transpose().clone();
                if i == j { prod[(0,0)] -= 1.0; }
                res.push( prod[(0,0)] );
            }
        }
        println!("{:?}", res);
        assert!( res.iter().all( |&x| x < ZTHRESH ) );
        println!("OK!");
    }

    #[test]
    fn rejection() {
        let y = na::DMatrix::from_row_slice(4, 9, &vec![
            -2.9559865737125,  2.5865278325864, -2.2980562996102,  0.7369790120865,
            -1.7022470264114,  2.5865278325864,  2.5865278325864, -0.7369790120865,
            -1.9907185593875,
             1.2508223406517, -0.7159569731051,  2.9493672391056, -3.8027781425851,
             3.6435053746187, -0.7159569731051, -0.7159569731051,  3.8027781425851,
             1.4100951086181,
            -0.9635641883542, -1.4738581147596, -2.9837682836541, -1.3922099258609,
            -2.0144903304535, -1.4738581147596, -1.4738581147596,  1.3922099258609,
             2.4431360679602,
            -0.8759863225138, -0.8875888657254,  2.6402078223407, -1.6158556098730,
             1.2462441219873, -0.8875888657254, -0.8875888657254,  1.6158556098730,
            -0.5063748346281]);

        let mut fs = FeasibleRegion::new( &y, Some(ZTHRESH) );

        fs.insert( 0, 0 );
        fs.insert( 0, 2 );
        fs.insert( 2, 2 );
        fs.insert( 2, 7 );

        let v = na::DMatrix::from_row_slice( 4, 4, &vec![
            1.0,  1.0,  1.0,  1.0,
            1.0,  1.0, -1.0, -1.0,
            1.0, -1.0,  1.0, -1.0,
            1.0, -1.0, -1.0,  1.0] );

        let res = fs.reject_mtx( &v );

        let p = fs.get_p();
        let mut residual_norms: Vec<f64> = Vec::new();

        for i in 0 .. 4 {
            for j in 0 .. p[i].len() {
                let residuals = res.clone() * p[i][j].transpose();
                for norm in residuals.iter() {
                    if *norm < ZTHRESH {
                        residual_norms.push( norm.clone() );
                    }
                }
            }
        }

        assert!(residual_norms.len() > 3 );
    }
}
