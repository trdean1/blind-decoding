extern crate nalgebra as na;

use std::fmt;

use ZTHRESH;

pub struct FeasibleRegion {
    y: na::DMatrix<f64>,
    dims: (usize,usize),
    b: Vec<na::DMatrix<f64>>,
    p: Vec<na::DMatrix<f64>>,
    col_map: Vec<Vec<usize>>,
    zthresh: f64,
}

impl Default for FeasibleRegion {
    fn default() -> FeasibleRegion {
        FeasibleRegion {
            y: na::DMatrix::from_column_slice(0,0, &Vec::new()),
            dims: (0,0),
            b: vec![na::DMatrix::from_column_slice(0,0, &Vec::new()); 1],
            p: vec![na::DMatrix::from_column_slice(0,0, &Vec::new()); 1],
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
            s += &format!("{:.4}\n", self.b[i]);
        }

        s += "p = \n";
        for i in 0 .. self.p.len() {
            s += &format!("{:.4}\n", self.p[i]);
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
            b: vec![na::DMatrix::from_column_slice(0,n, &Vec::new()); n],
            p: vec![na::DMatrix::from_column_slice(0,n, &Vec::new()); n],
            col_map: vec![Vec::new(); n],
            zthresh: match zthresh{ Some(z) => z, None => ZTHRESH },

            ..Default::default()
        }
    }
}
