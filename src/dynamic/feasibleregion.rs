extern crate nalgebra as na;

use std::fmt;

use ZTHRESH;

pub struct FeasibleRegion {
    y: na::DMatrix<f64>,
    dims: (usize,usize),
    b: Vec<Vec<na::RowDVector<f64>>>,
    p: Vec<Vec<na::RowDVector<f64>>>,
    col_map: Vec<Vec<usize>>,
    zthresh: f64,
}

impl Default for FeasibleRegion {
    fn default() -> FeasibleRegion {
        FeasibleRegion {
            y: na::DMatrix::from_column_slice(0,0, &Vec::new()),
            dims: (0,0),
            b: vec![vec![na::RowDVector::from_column_slice(0, &Vec::new()); 1]],
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

        s += "b = \n";
        for i in 0 .. self.b.len() {
            for j in 0 .. self.b[i].len() {
                s += &format!("{:.4}\n", self.b[i][j]);
            }
        }

        s += "p = \n";
        for i in 0 .. self.p.len() {
            for j in 0 .. self.p[i].len() {
                s += &format!("{:.4}\n", self.p[i][j]);
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
            b: vec![Vec::with_capacity(k); n],
            p: vec![Vec::with_capacity(n); n],
            col_map: vec![Vec::with_capacity(n); n],
            zthresh: match zthresh{ Some(z) => z, None => ZTHRESH },

            ..Default::default()
        }
    }

    pub fn insert_mtx( &mut self, update: &na::DMatrix<bool> ) {
        for i in 0 .. update.nrows() {
            for j in 0 .. update.ncols() {
                if update[(i,j)] {
                    self.insert( i, j );
                }
            }
        }
    }

    pub fn insert( &mut self, row: usize, column: usize ) {
        //Check to see if we have already added this entry
        if self.col_map[row].iter().any( |&x| x == column ) {
            return;
        }

        //Copy column of y and insert into b
        self.b[row].push( self.y.column(column).transpose().into_owned() );

        //Update column map
        self.col_map[row].push(column);

        //Update p
        let mut v: na::RowDVector<f64> = 
            self.y.column(column).transpose().into_owned();

        let mut v_norm = na::DMatrix::from_column_slice(1,1,&vec![0.0;1]);
        let mut u_norm = na::DMatrix::from_column_slice(1,1,&vec![0.0;1]);
        let mut uv = na::DMatrix::from_column_slice(1,1,&vec![0.0;1]);
        
        //Normalize v
        v.mul_to( &v.transpose(), &mut v_norm );
        v /= v_norm[(0,0)].sqrt();

        //Perform matrix rejection, add v to p as long as it is not orthogonal
        for i in 0 .. self.p[row].len() {
            let u = &self.p[row][i];

            //uv = (u * v.T)
            v.mul_to( &u, &mut uv );

            //u_norm = u * u.T
            u.mul_to( &u.transpose().clone(), &mut u_norm );

            //v = v - ((u * v.T) / (u * u.T)) * u
            v -= (uv[(0,0)] / u_norm[(0,0)]) * u.clone();

            //Check if v is orthogonal to u
            v.mul_to( &v.transpose(), &mut v_norm );
            if v_norm[(0,0)].sqrt() < self.zthresh { return }

            //Normalize and continue
            v /= v_norm[(0,0)].sqrt();
        }
        
        //If p already has n entries then we should have returned already
        assert!( self.p[row].len() < self.dims.0 );
        self.p[row].push( v );
    }

    pub fn reject_mtx( &self, v: na::DMatrix<f64> )
            -> na::DMatrix<f64> {
        let n = v.ncols();
        let mut vv: na::DMatrix<f64> = na::DMatrix::from_column_slice(0,n,&Vec::new());

        for i in 0 .. v.nrows() {
            let v_row = v.row( i );
            match self.reject_vec( &v_row.transpose(), i ) {
                Some(r) => {
                    vv = vv.insert_row( i, 0.0 );
                    for j in 0 .. n {
                        vv[(i,j)] = r[(j,0)];
                    }
                }
                None => {
                    vv = vv.insert_row( i, 0.0 );
                },
            }
        }

        vv
    }

    pub fn reject_vec( &self, v: &na::DVector<f64>, row: usize ) 
            -> Option<na::DVector<f64>> {
        let mut vv = v.transpose().into_owned();

        let mut uu = na::DMatrix::from_column_slice(1,1,&vec![0.0;1]);
        let mut uv = na::DMatrix::from_column_slice(1,1,&vec![0.0;1]);

        for i in 0 .. self.p[row].len() {
            let u = &self.p[row][i];
            vv.mul_to( &u.transpose(), &mut uv ); 
            u.mul_to( &u.transpose(), &mut uu );

            if uu[(0,0)].sqrt() < self.zthresh{ return None; }

            vv -= (uv[(0,0)] / uu[(0,0)]) * u;
        }
        
        Some(vv.transpose())
    }

}
