extern crate nalgebra as na;

use std;
use std::fmt;
use std::error;
use std::error::Error;
use std::collections::{HashSet};

use ZTHRESH;
use equal_atm;

#[allow(unused)]
const VERBOSE_HOP: u64 = 0x1;
#[allow(unused)]
const VERBOSE_BFS: u64 = 0x2;
#[allow(unused)]
const VERBOSE_CONSTR_CREATE: u64 = 0x4;
#[allow(unused)]
const VERBOSE_TAB_MAP: u64 = 0x8;
#[allow(unused)]
const VERBOSE_EXTRA_CONSTR: u64 = 0x10;
#[allow(unused)]
const VERBOSE_EVAL_VERTEX: u64 = 0x20;
#[allow(unused)]
const VERBOSE_FLIP_GRADIENT: u64 = 0x40;
#[allow(unused)]
const VERBOSE_FLIP: u64 = 0x80;
#[allow(unused)]
const VERBOSE_INITIAL_TAB: u64 = 0x100;
#[allow(unused)]
const VERBOSE_INDEP: u64 = 0x10000;
#[allow(unused)]
const VERBOSE_VERTEX: u64 = 0x20000;
#[allow(unused)]
const VERBOSE_ALL: u64 = 0xFFFFFFFF;

const VERBOSE: u64 = 0;

// Tableau{@
#[derive(Debug)] //{@
/// Error type for FlexTab.
//@}
pub enum FlexTabError { //{@
    SingularInput,
    GoodCols,
    LinIndep,
    NumZeroSlackvars,
    URowRedef,
    XVarsBothNonzero,
    StateStackExhausted,
    Trapped,
    TooManyHops,
    Runout,
} //@}
impl error::Error for FlexTabError { //{@
    fn description(&self) -> &str {
        match self {
            &FlexTabError::SingularInput => "Input Y is not full rank",
            &FlexTabError::LinIndep => "Insuffient linearly independent columns",
            &FlexTabError::GoodCols => "Insufficient good columns",
            &FlexTabError::NumZeroSlackvars => "Num zero slackvars != 0",
            &FlexTabError::URowRedef => "U row redefinition",
            &FlexTabError::XVarsBothNonzero => "Xvars both nonzero",
            &FlexTabError::StateStackExhausted => "State stack exhausted",
            &FlexTabError::Trapped => "Found trap case",
            &FlexTabError::TooManyHops => "Too many hops",
            &FlexTabError::Runout => "Ran out of attempts", 
        }
    }
} //@}
impl fmt::Display for FlexTabError { //{@
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }
} //@}

#[derive(Clone)] //{@
/// Structure to maintain state during vertex hopping to support backtracking.
//@}
pub struct State { //{@
    x: Vec<f64>,
    u: na::DMatrix<f64>,
    uy: na::DMatrix<f64>,
    uybad: Option<na::DMatrix<f64>>,

    rows: na::DMatrix<f64>,
    vmap: Vec<usize>,

    grad: na::DMatrix<f64>,
    flip_grad: Vec<f64>,
    flip_grad_max: f64,
    flip_grad_min: f64,

    obj: f64,
    det_u: f64,
    vertex: Vertex,
} //@}
impl Default for State { //{@
    fn default() -> State {
        State {
            x: Vec::new(),
            u: na::DMatrix::from_column_slice(0, 0, &Vec::new()),
            uy: na::DMatrix::from_column_slice(0, 0, &Vec::new()),
            uybad: None,
            
            rows: na::DMatrix::from_column_slice(0, 0, &Vec::new()),
            vmap: Vec::new(),

            grad: na::DMatrix::from_column_slice(0, 0, &Vec::new()),
            flip_grad: Vec::new(),
            flip_grad_max: 0.0,
            flip_grad_min: 0.0,

            obj: 0.0,
            det_u: 0.0,
            vertex: Vertex::new(0),
        }
    }
} //@}
impl State { //{@
    fn new(u: na::DMatrix<f64>, uy: na::DMatrix<f64>, //{@
            uybad: Option<na::DMatrix<f64>>) -> State {
        let (n, k) = uy.shape();
        State {
            x: vec![0.0; 2 * (n*n + n*k)], 
            u: u,
            uy: uy,
            uybad: uybad,

            rows: na::DMatrix::from_column_slice(2 * n*k, 2 * (n*n + n*k) + 1,
                    &vec![0.0; (2 * n*k) * (2 * (n*n + n*k) + 1)]),
            vmap: vec![0; n*n],

            grad: na::DMatrix::from_column_slice(n, n, &vec![0.0; n*n]),
            flip_grad: vec![0.0; n*n],

            vertex: Vertex::new(n * n),

            ..Default::default()
        }
    } //@}

    fn copy_from(&mut self, other: &State) { //{@
        self.x.copy_from_slice(&other.x);
        self.vmap.copy_from_slice(&other.vmap);
        self.rows.copy_from(&other.rows);
        self.u.copy_from(&other.u);
        self.grad.copy_from(&other.grad);
        self.flip_grad.copy_from_slice(&other.flip_grad);
        self.uy.copy_from(&other.uy);
        match self.uybad {
            None => (),
            Some(ref mut uybad) => match other.uybad {
                Some(ref other_uybad) => uybad.copy_from(&other_uybad),
                None => panic!("Attempted copy from empty state uybad"),
            },
        }
        self.obj = other.obj;
        self.det_u = other.det_u;
        self.vertex.copy_from(&other.vertex);
    } //@}

    pub fn get_u(&self) -> na::DMatrix<f64> {
        self.u.clone()
    }

    pub fn get_uy(&self) -> na::DMatrix<f64> {
        self.uy.clone()
    }

    pub fn uy_equal_atm(&self, x: &na::DMatrix<f64>)
        -> bool {
        equal_atm(&self.uy, x)
    }

    pub fn uy_is_pm1(&self, zthresh: f64) -> bool {
        self.uy.iter().all(|&e| (e.abs() - 1.0).abs() < zthresh)
    }

    pub fn obj(&self) -> f64 {
        return self.obj;
    }
} //@}

#[derive(Hash, Eq, Clone)] //{@
/// Structure for capturing vertices as a simple bool vector for use in
/// HashSets.
//@}
struct Vertex { //{@
    bits: Vec<bool>,
} //@}
impl Vertex { //{@
    fn new(nsq: usize) -> Vertex {
        Vertex { bits: vec![false; nsq] }
    }

    #[allow(dead_code)]
    fn from_vmap(vmap: &[usize]) -> Vertex {
        let mut vertex = Vertex::new(vmap.len());
        for (i, &v) in vmap.iter().enumerate() {
            vertex.bits[i] = v & 0x1 == 1;
        }
        vertex
    }
    
    fn copy_from(&mut self, other: &Vertex) {
        self.bits.copy_from_slice(&other.bits)
    }

    fn flip(&mut self, idx: usize) {
        self.bits[idx] = !self.bits[idx];
    }
} //@}
impl PartialEq for Vertex { //{@
    fn eq(&self, other: &Vertex) -> bool {
        if self.bits.len() != other.bits.len() {
            return false;
        }

        for i in 0 .. self.bits.len() {
            if self.bits[i] != other.bits[i] {
                return false;
            }
        }

        true
    }
} //@}

#[derive(Clone)] //{@
/// Structure for constraints that arise from extra "good" columns of Y beyond
/// the first n linearly independent columns.  These are referred to as type 3
/// constraints.
//@}
struct Constraint { //{@
    addends: Vec<(usize, f64)>,
    sum: f64,
}
impl Constraint { //{@
    fn new() -> Constraint {
        Constraint { addends: Vec::new(), sum: 0.0 }
    }

    fn push_addend(&mut self, xvar: usize, coeff: f64) {
        self.addends.push((xvar, coeff));
    }

    /// Check if the constraint is still satisfied.
    fn check(&self, x: &Vec<f64>, zthresh: f64) -> bool {
        if self.addends.len() == 0 { return true; }

        let total = self.addends.iter().fold(0.0, |total, &(xvar, coeff)| {
            total + coeff * x[xvar]
        });
        debug!("Difference: {}", (total-self.sum).abs());
        (total - self.sum).abs() < zthresh
    }
} //@}
impl Eq for Constraint {} //{@
impl PartialEq for Constraint {
    fn eq(&self, other: &Constraint) -> bool {
        if (self.sum - other.sum).abs() > ZTHRESH {
            return false;
        }

        if self.addends.len() != other.addends.len() {
            return false;
        }

        // Sort the addends by xvar.
        let mut s_addends = self.addends.clone();
        let mut o_addends = other.addends.clone();
        s_addends.sort_by_key(|&a| a.0);
        o_addends.sort_by_key(|&a| a.0);
        
        // Check the sorted addends.
        for i in 0 .. s_addends.len() {
            let ref s_add = s_addends[i];
            let ref o_add = o_addends[i];
            if s_add.0 != o_add.0 || (s_add.1 - o_add.1).abs() > ZTHRESH {
                return false;
           }
        }
        true
    }
} //@}
impl fmt::Display for Constraint { //{@
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} = {}",
                self.addends.iter().map(|&(xvar, coeff)|
                        format!("{}x_{}", coeff, xvar))
                    .collect::<Vec<_>>()
                    .join("  +  "),
                self.sum)
    }
} //@}
//@}

//{@
/// Principal tableau structure.
//@}
pub struct FlexTab { //{@
    verbose: u64,
    zthresh: f64,

    n: usize,
    k: usize,

    y: na::DMatrix<f64>,
    ybad: Option<na::DMatrix<f64>>,

    urows: Vec<Option<(usize, usize, usize)>>,
    extra_constr: Vec<Vec<Constraint>>,

    pub state: State,
    pub best_state: State,

    visited: HashSet<Vertex>,
    statestack: Vec<State>,
} //@}
impl Default for FlexTab { //{@
    fn default() -> FlexTab {
        FlexTab {
            verbose: 0,
            zthresh: 1e-9,

            n: 0,
            k: 0,

            y: na::DMatrix::from_column_slice(0, 0, &Vec::new()),
            ybad: None,

            urows: Vec::new(),
            extra_constr: Vec::new(),

            state: State { ..Default::default() },
            best_state: State { ..Default::default() },

            visited: HashSet::new(),
            statestack: Vec::new(),
        }
    }
} //@}
impl fmt::Display for FlexTab { //{@
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();

        // X-Values: 4 per line.
        for (i, &elt) in self.state.x.iter().enumerate() {
            s += &format!("x_{} = {:5.5}{}",
                    i, elt, if i % 4 == 3 { "\n" } else { " " });
        }
        s += &format!("\n");

        // Constraints: 1 per line.
        for i in 0 .. self.state.rows.nrows() {
            let r = self.state.rows.row(i);
            let mut line = Vec::new();
            for j in 0 .. r.len() - 1 {
                let v = r[j];
                if v.abs() > self.zthresh {
                    line.push(format!("{:5.2}x_{}", v, j));
                }
            }
            let line = line.join(" + ");
            s += &format!("{} == {:5.2}\n", line, r[r.len() - 1]);
        }

        // Type 3 constraints.
        println!("Type 3 Constraints:");
        for (rownum, ref cset) in self.extra_constr.iter().enumerate() {
            println!("\trow: {}", rownum);
            for ref c in cset.iter() {
                println!("\t\t{}", c);
            }
        }
        println!("\n");

        // urows
        s += "\nurows:\n";
        for (i, &u) in self.urows.iter().enumerate() {
            if let Some(u) = u {
                s += &format!("{:2} => {:?}", i, u);
            } else {
                s += &format!("{:2} => None", i);
            }
            s += if i % self.n == self.n - 1 { "\n" } else { " | " };
        }

        // vmap
        s += "\nvmap:\n";
        for (i, &v) in self.state.vmap.iter().enumerate() {
            s += &format!("{:2} => {:2}", i, v);
            s += if i % self.n == self.n - 1 { "\n" } else { " | " };
        }

        if let Some(ref y) = self.ybad {
            s += &format!("uybad = {:.5}\n", 
                          self.state.u.clone() * y.clone());
        }

        // current u, y, uy
        s += &format!("u = {:.5}y = {:.5}uy = {:.5}\n", self.state.u, self.y,
                self.state.uy);
        
        write!(f, "{}", s)
    }
} //@}
impl FlexTab { //{@
    pub fn new(u: &na::DMatrix<f64>, y: &na::DMatrix<f64>, zthresh: f64) //{@
            -> Result<FlexTab, FlexTabError> { 
        // Find all bad columns.
        let (bad_cols, uyfull) = FlexTab::find_bad_columns(&u, &y, zthresh);
        debug!("Bad Columns: {:?}", bad_cols);
        let n = y.nrows();
        let k = y.ncols() - bad_cols.len();

        // Ensure that there are at least n good columns; later we will check
        // for linear independence during tableau creation.
        if k < n {
            return Err(FlexTabError::GoodCols);
        } 

        //let mut uy = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
        let mut uy: na::DMatrix<f64>;
        let mut y_used: na::DMatrix<f64>;
        let mut ybad = None;
        let mut uybad = None;

        if bad_cols.len() == 0 {
            // All columns are good, so we use them all.
            y_used = y.clone();
            uy = uyfull;

        } else {
            // Some columns are bad: only used for feasibility restrictions.
            y_used = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
            uy = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
            let mut _ybad = na::DMatrix::from_column_slice(n, bad_cols.len(),
                    &vec![0.0; n * bad_cols.len()]);
            let mut _uybad = na::DMatrix::from_column_slice(n, bad_cols.len(),
                    &vec![0.0; n * bad_cols.len()]);

            let mut cur_good = 0;
            let mut cur_bad = 0;
            for j in 0 .. y.ncols() {
                let src_col = y.column(j);
                if bad_cols.iter().any(|&bc| bc == j) {
                    _ybad.column_mut(cur_bad).copy_from(&src_col);
                    _uybad.column_mut(cur_bad).copy_from(&uyfull.column(j));
                    cur_bad += 1;
                } else {
                    y_used.column_mut(cur_good).copy_from(&src_col);
                    uy.column_mut(cur_good).copy_from(&uyfull.column(j));
                    cur_good += 1;
                }
            }
            assert!(cur_bad == bad_cols.len() && cur_good == k);

            ybad = Some(_ybad);
            uybad = Some(_uybad);
        }

        u.mul_to(&y_used, &mut uy);

        // Initialize mutable state.
        let state = State::new(u.clone(), uy, uybad);
        let best_state = state.clone();

        let mut ft = FlexTab {
            zthresh: zthresh,
            verbose: VERBOSE,
            
            n: n,
            k: k,

            y: y_used,
            ybad: ybad,

            urows: vec![None; n * n],
            extra_constr: vec![Vec::new(); n],

            state: state,
            best_state: best_state,

            ..Default::default()
        };
        ft.initialize_x();
        ft.set_constraints();
        Ok(ft)
    } //@}
    //{@
    /// Find the indices of all +bad+ columns, meaning there is at least one
    /// entry that is not \pm 1.
    /// Output: 1. Vector of indices of bad columns.
    ///         2. Product of UY for the full Y matrix.
    //@}
    fn find_bad_columns(u: &na::DMatrix<f64>, y: &na::DMatrix<f64>, //{@
            zthresh: f64) -> (Vec<usize>, na::DMatrix<f64>) {
        let (n, k) = y.shape();
        let mut bad_cols = Vec::new();
        let mut uyfull = na::DMatrix::from_column_slice(n, k, &vec![0.0; n*k]);
        u.mul_to(&y, &mut uyfull);
        for j in 0 .. uyfull.ncols() {
            if uyfull.column(j).iter().any(|&e| (e.abs() - 1.0).abs() > zthresh) {
                bad_cols.push(j);
            }
        }
        (bad_cols, uyfull)
    } //@}
    //{@
    /// Number of columns such that uy = \pm 1.  This is NOT updated but
    /// reflects solely the original value.
    /// TODO: This is really dead code since y is only good columns
    //@}
    #[allow(dead_code)]
    pub fn num_good_cols(&self) -> usize { //{@
        self.y.ncols()
        /*
        match self.ybad {
            Some(ref ybad) => self.y.ncols(),// - ybad.ncols(),
            None => self.y.ncols(),
        }
        */
    } //@}
    #[allow(dead_code)]
    fn num_indep_cols(&self) -> usize { //{@
        // Determine the rank of Y.
        let svd = na::SVD::new(self.y.clone(), false, false);
        svd.singular_values.iter().filter(|&elt| *elt > self.zthresh).count()
    } //@}
    fn initialize_x(&mut self) { //{@
         for j in 0 .. self.state.u.ncols() {
             for i in 0 .. self.state.u.nrows() {
                 let idx = 2 * (self.n * i + j);
                 let elt = self.state.u.column(j)[i];
                 if elt >= 0.0 {
                     self.state.x[idx] = elt;
                 } else {
                     self.state.x[idx + 1] = -elt;
                 }

             }
         }
    } //@}
    //{@
    /// Set the constraints that will be used for vertex hoppinng.  These are
    /// only the constraints imposed by self.y, thus only the +good+ columns of
    /// Y.  The other constraints, imposed by the +bad+ columns of Y, are
    /// captured in self.extra_constr.
    //@}
    fn set_constraints(&mut self) { //{@
        let mut constraint = vec![(0, 0.0); self.n];
        let mut tab_row = 0;

        for i in 0 .. self.n {
            let x_var_base = i * self.n;
            for j in 0 .. self.k {
                {
                    let col = self.y.column(j);
                    // The i^th row uses variables in range: i*n .. i*(n+1).
                    // We are constraining |Uy|_\infty <= 1.
                    // Constraints are n-vector fo tuples [(varnum, coeff), ...  ].
                    for q in 0 .. self.n {
                        constraint[q] = (x_var_base + q, col[q]);
                    }
                }
                self.add_constraint_pair(tab_row, &constraint);
                tab_row += 2;
            }
        }
    } //@}
    //{@
    /// Add 2 new constraints based on the content of cset, which should be a
    /// slice of 2-tuples of the form (varnum: usize, coeff: f64).
    //@}
    fn add_constraint_pair(&mut self, tab_row: usize, cset: &[(usize, f64)]) { //{@
        // Setup the two new constraints based on the vars / coeffs given.
        for &(var, coeff) in cset.iter() {
            // Example: 3.1x_0 will become:
            //     tab_row    :  3.1x'_0 - 3.1x'_1 <= 1.
            //     tab_row +1 : -3.1x'_0 + 3.1x'_1 <= 1.
            self.state.rows.row_mut(tab_row)[2 * var] = coeff;
            self.state.rows.row_mut(tab_row)[2 * var + 1] = -coeff;
            self.state.rows.row_mut(tab_row + 1)[2 * var] = -coeff;
            self.state.rows.row_mut(tab_row + 1)[2 * var + 1] = coeff;
        }

        // Slack var coeffs both = 1.0.
        let zeroth_slack = 2 * self.n * self.n + tab_row;
        let first_slack = zeroth_slack + 1;
        self.state.rows.row_mut(tab_row)[zeroth_slack] = 1.0;
        self.state.rows.row_mut(tab_row + 1)[first_slack] = 1.0;

        // RHS of both constraints = 1.0.
        let rhs = self.state.rows.ncols() - 1;
        self.state.rows.row_mut(tab_row)[rhs] = 1.0;
        self.state.rows.row_mut(tab_row + 1)[rhs] = 1.0;

        // Need to determine values of the slack variables for this pair of
        // constraints.  Compute the value of the LHS of the constraint.  One of
        // the pair will be tight, so slack will be 0; other slack will be 2.
        // Compute: val = \sum_{2n LHS vars in this constr_set} a_{ij} * x_j.
        let x_lowbound = (2 * self.n) * (tab_row / (2 * self.k));
        let x_highbound = x_lowbound + 2 * self.n;
        let val = (x_lowbound .. x_highbound).fold(0.0, |val, j|
                val + self.state.x[j] * self.state.rows.row(tab_row)[j]);
        // Set slack var values so LHS = 1.0.
        self.state.x[zeroth_slack] = 1.0 - val;
        self.state.x[first_slack] = 1.0 + val;
    } //@}

    //{@
    /// Create simplex-style tableau, which will be stored in self.rows.  Note
    /// that self.X is of length 2 * self.nk.  There are 2 * self.nk overall
    /// constraints in |UY|_\infty \leq 1.
    //@}
    fn to_simplex_form(&mut self) -> Result<(), FlexTabError> { //{@
        // Zero out any entries in self.x that are solely float imprecision.
        let num_x_from_u = 2 * self.n * self.n;

        // There are n groups, each of k equation pairs, and within a group
        // every equation pair uses the same set of x variables.
        for groupnum in 0 .. self.n {
            // Grab each [of the k] equation pairs individually.  Idea is to
            // iterate over all still-available basic vars, then if one exists
            // such that self.rows[row, var] > zthresh, then use var as the
            // basic var for this eq_set [where row = 2 * eq_set].
            let mut avail_basic = (groupnum * self.n .. (groupnum + 1) * self.n)
                .collect::<HashSet<_>>();

            for j in 0 .. self.k {
                // eq_pair = pair of equations we are currently focused on.
                let eq_pair = groupnum * self.k + j;

                // Exactly one of eq_pair should have a zero slack variable.
                let lowzero = self.state.x[num_x_from_u + 2 * eq_pair].abs()
                    < self.zthresh;
                let highzero = self.state.x[num_x_from_u + 2 * eq_pair + 1].abs()
                    < self.zthresh;
                if !(lowzero ^ highzero) {
                    info!("num zero slackvars in pair {} != 1", eq_pair);
                    return Err(FlexTabError::NumZeroSlackvars);
                }
                let zeroslack = if lowzero { 2 * eq_pair } else { 2 * eq_pair + 1 };

                // Find available basic vars with nonzero coeff in this pair.
                let uvar = {
                    let useable = avail_basic.iter()
                        .filter(|&v| self.state.rows.row(2 * eq_pair)[2 * v].abs()
                                > self.zthresh)
                        .collect::<Vec<_>>();
                    match useable.first() {
                        Some(uvar) => Some(**uvar),
                        None => None,
                    }
                };
                        
                // If we have an available uvar then set a basic variable;
                // otherwise, simply add one row to the other.
                if let Some(uvar) = uvar {
                    // Use the row where slack = 0 to turn an original U-var
                    // into a basic var.  For ease use the original x_j, so we
                    // make basic var from whichever of x_{2j}, x_{2j+1} != 0.
                    avail_basic.remove(&uvar);
                    let x_basic = zeroslack;
                    self.set_basic(uvar, x_basic)?;
                } else {
                    let tgtrow = zeroslack;
                    let srcrow = tgtrow ^ 0x1;
                    self.add_row_multiple(tgtrow, srcrow, 1.0);
                }
            }
        }
        self.tableau_mappings()?;
        Ok(())
    } //@}
    //{@
    /// Create a simplex-style basic variable from the original x_j.  Note that
    /// the original x_j \in R is now divided into the difference of variables
    /// \in R^+: x_{2j} and x_{2j+1}.  We make a basic variable from whichever
    /// of x_{2j}, x_{2j+1} is nonzero.  The selected variable has its
    /// coefficient in pivot_row set to 1, and zeroed everywhere else.
    /// INPUT:  j = original x-var: the new basic var will be whichever of
    ///             x_{2j}, x_{2j+1} is nonzero.
    ///         pivot_row = row in which the selected new basic var should have
    ///             coefficient set to 1.
    /// OUTPUT: true on success.
    //@}
    fn set_basic(&mut self, j: usize, pivot_row: usize) //{@
            -> Result<(), FlexTabError> {
        // basic var = whichever of x_{2j}, x_{2j+1} != 0.
        let lownz = self.state.x[2*j].abs() > self.zthresh;
        let highnz = self.state.x[2*j + 1].abs() > self.zthresh;
        if lownz && highnz {
            error!("xvars {}, {} both != 0", 2*j, 2*j + 1);
            return Err(FlexTabError::XVarsBothNonzero);
        }
        let tgtvar = if lownz { 2 * j } else { 2 * j + 1 };

        // Make coeff of pivot_row[tgtvar] = 1.
        let divisor = self.state.rows.row(pivot_row)[tgtvar];
        self.div_row_float(pivot_row, divisor);

        // Eliminate tgtvar from every other row.
        let baserow = 2 * (j / self.n * self.k);
        let limrow = baserow + 2 * self.k;
        for i in baserow .. limrow {
            if i == pivot_row { continue; }
            let mult = -1.0 * self.state.rows.row(i)[tgtvar];
            self.add_row_multiple(i, pivot_row, mult);
        }

        Ok(())
    } //@}
    fn tableau_mappings(&mut self) -> Result<(), FlexTabError> { //{@
        // Each constraint set corresponds to one row of U.
        for cset in 0 .. self.n {
            let mut vvars:HashSet<usize> = HashSet::new();
            for row in 2 * self.k * cset .. 2 * self.k * (cset + 1) {
                let (uvars, slackvars) = self.categorize_row(row);

                // If there are any uvars, there must be 2, both from same Xvar.
                if uvars.len() > 0 {
                    if uvars.len() != 2 || uvars[0] >> 1 != uvars[1] >> 1 {
                        // This fails if not enough lin indep cols.
                        return Err(FlexTabError::LinIndep);
                    }
                    let uvar = uvars[0] >> 1;
                    match self.urows[uvar] {
                        Some(_) => {
                            error!("urows[{}] redef in row {}", uvar, row);
                            return Err(FlexTabError::URowRedef);
                        },
                        None => {
                            // Tuple is (row, xplus, xminus), where +row+ is the
                            // actual tableau row, +xplus+ is the tableau column
                            // in which the variable +uvar+ has a + coeff.
                            self.urows[uvar] = 
                                if self.state.rows.row(row)[2 * uvar] > 0.0 {
                                    Some((row, 2 * uvar, 2 * uvar + 1))
                                } else {
                                    Some((row, 2 * uvar + 1, 2 * uvar))
                                };
                        }
                    };
                    // All of the RHS vars in 2-uvar eqns are "normal" slacks.
                    for v in slackvars { vvars.insert(v); }

                } else {
                    // If exactly 2 slackvars and both correspond to same base
                    // eqn, then this is dealt with in vvars.  Otherwise, this
                    // gives us a type 3 constraint.
                    let x_u_count = 2 * self.n * self.n;
                    let slack_eqn = |q: usize| (q - x_u_count) >> 1;
                    if slackvars.len() != 2
                          || slack_eqn(slackvars[0]) != slack_eqn(slackvars[1]) {
                        self.add_extra_constr(cset, row, &slackvars);
                    }
                }
            }

            let mut vvars = vvars.iter().map(|&v| v).collect::<Vec<usize>>();
            vvars.sort();
            for (i, &v) in vvars.iter().enumerate() {
                self.state.vmap[cset * self.n + i] = v;
            }
        }
        Ok(())
    } //@}
    //{@
    /// Return arrays of variables that are:
    ///  1. uvars: X vars derived from U vars (potential basic)
    ///  2. slackvars: X vars that are slackvars
    //@}
    fn categorize_row(&self, rownum: usize) -> (Vec<usize>, Vec<usize>) { //{@
        // Determine which series we are in, where the series is the row of U
        // from which these constraints are derived.
        let g = rownum / (2 * self.k);
        let row = self.state.rows.row(rownum);

        // The 2n U-vars are those from the relevant row.
        let low_u = 2 * self.n * g;
        let high_u = low_u + 2 * self.n;
        let uvars = (low_u .. high_u)
            .filter(|&j| row[j].abs() > self.zthresh)
            .collect::<Vec<_>>();

        // The 2k potential slack vars are those involving these same U rows.
        let low_s = (2 * self.n * self.n) + (2 * self.k * g);
        let high_s = low_s + 2 * self.k;
        let slackvars = (low_s .. high_s)
            .filter(|&j| row[j].abs() > self.zthresh)
            .collect::<Vec<_>>();

        (uvars, slackvars)
    } //@}
    fn add_extra_constr(&mut self, csetnum: usize, rownum: usize, //{@
            slackvars: &Vec<usize>) {
        let lastcol = self.state.rows.ncols() - 1;
        let row = self.state.rows.row(rownum);
        let sum = row[lastcol];

        // Create constraint, set sum.
        let mut con = Constraint::new();
        con.sum = sum;

        // Push all addends to this constraint.
        for &v in slackvars {
            con.push_addend(v, row[v]);
        }

        // Add fully formed constraint to relevant extra constraint set.
        self.extra_constr[csetnum].push(con);
    } //@}

    fn snapshot(&mut self) { //{@
        self.statestack.push(self.state.clone());
    } //@}
    fn restore(&mut self, pop: bool) -> bool { //{@
        if pop {
            if let Some(state) = self.statestack.pop() {
                self.state.copy_from(&state);
                true
            } else {
                false
            }
        } else {
            if let Some(state) = self.statestack.last() {
                self.state.copy_from(&state);
                true
            } else {
                false
            }
        }
    } //@}

    pub fn solve(&mut self) -> Result<(), FlexTabError> { //{@
        self.to_simplex_form()?;
        self.hop()?;
        Ok(())
    } //@}
    fn hop(&mut self) -> Result<(), FlexTabError> { //{@
        self.mark_visited();

        let mut extra_cols = 0;
        {
            match self.state.uybad {
                Some(ref uybad) => {
                    debug!("starting uybad = {:.4}", uybad);
                    extra_cols = uybad.ncols();
                },
                None => {},
            }
        }

        // Principal vertex hopping loop.
        loop {
            if self.verbose & VERBOSE_HOP != 0 {
                println!("HOP loop top: visited = {}", self.visited.len());
            }

            // Check if this vertex is valid: if not, backtrack.
            if !self.eval_vertex() {
                debug!("Found infeasible vertex");
                if !self.restore(true) {
                    return Err(FlexTabError::StateStackExhausted);
                }
                continue;
            } else {
                debug!("Found feasible vertex");
                match self.state.uybad {
                    Some(ref uybad) => debug!("uybad = {:.4}", uybad),
                    None => {},
                }
            }
            // Check if this is the best vertex we have seen.
            if self.state.obj > self.best_state.obj {
                self.best_state.copy_from(&self.state);
            }
            // Check if this vertex is a global optimum.
            if self.is_done() {
                break;
            }

            if self.visited.len() >= 2 * self.n * ( self.k + extra_cols ) {
                return Err(FlexTabError::TooManyHops);
            }
            let (flip_idx, effect) = self.search_neighbors();
            match flip_idx {
                None => {
                    if !self.restore(true) {
                        return Err(FlexTabError::StateStackExhausted);
                    }
                },
                Some(idx) => {
                    // Take snapshot, flip idx, mark new vertex visited.
                    self.snapshot();
                    if self.verbose & VERBOSE_HOP != 0 {
                        println!("Hop {}", if effect > self.zthresh
                                { "++" } else { "==" });
                    }
                    self.flip(idx);
                    self.mark_visited();
                }
            };
        }
        Ok(())
    } //@}
    fn search_neighbors(&mut self) -> (Option<usize>, f64) { //{@
        let mut best_idx = None;
        let mut best_effect = std::f64::MIN;

        //debug!("Effects:");
        for i in 0 .. self.n {
            for j in 0 .. self.n {
                let v = self.n * i + j;
                let effect = self.state.flip_grad[v];
                //debug!("({},{}): {:.5} (Visited: {})", i, j, effect,
                //        self.is_flip_visited(v));
                if !self.is_flip_visited(v)
                        && !effect.is_infinite()
                        && effect > best_effect {
                    best_idx = Some(v);
                    best_effect = effect;
                }
            }
        }
        if best_idx != None {
            debug!("Best: ({}), {:.4}\n", best_idx.unwrap(), best_effect);
        } else {
            debug!("No good neighbor found\n");
        }
        (best_idx, best_effect)
    } //@}

    //{@
    /// Flip the variable U_v between {-1, +1}.  Since the variable U_v is
    /// represented as the difference X_{2v} - X_{2v+1}, this amounts to
    /// changing which of these variables is basic, so the variable that
    /// currently has value 0 will be the entering variable, and the one with
    /// value 2 will be the leaving variable.
    /// We are flipping the value of UY at flat index v, where flat indices are
    /// in row-major order.
    //@}
    fn flip(&mut self, v: usize) { //{@
        // Set vertex used for visited hash.
        self.state.vertex.flip(v);
        
        // The self.vmap holds the index v such that x_v is the currently 0
        // slack variable that needs to be flipped to 2 in order to exec pivot.
        let old_zero = self.state.vmap[v];
        let new_zero = old_zero ^ 0x1;
        self.state.vmap[v] = new_zero;
        self.state.x[old_zero] = 2.0;
        self.state.x[new_zero] = 0.0;

        // To exec the pivot, we must pivot each of the (n) u-values.
        let base = (v / self.n) * self.n;
        let top = base + self.n;
        for i in base .. top {
            let (maniprow, xplus, xminus) = self.urows[i].unwrap();
            if self.verbose & VERBOSE_FLIP != 0 {
                println!("i = {}, maniprow = {}", i, maniprow);
            }

            // Each maniprow will have the old_zero variable: get coefficient.
            let old_zero_coeff = self.state.rows.row(maniprow)[old_zero];

            // Execute pivot.
            let mut row = self.state.rows.row_mut(maniprow);
            let rhs = row.len() - 1;
            row[rhs] -= old_zero_coeff * 2.0;
            row[old_zero] = 0.0;
            row[new_zero] = -old_zero_coeff;

            // Set the relevant value of x.
            if row[rhs] >= 0.0 {
                self.state.x[xplus] = row[rhs];
                self.state.x[xminus] = 0.0;
            } else {
                self.state.x[xplus] = 0.0;
                self.state.x[xminus] = -row[rhs];
            }
        }
    } //@}
    fn is_flip_visited(&mut self, idx: usize) -> bool { //{@
        self.state.vertex.flip(idx);
        let ans = self.visited.contains(&self.state.vertex);
        self.state.vertex.flip(idx);
        ans
    } //@}
    fn eval_vertex(&self) -> bool { //{@
        if self.state.obj < -10.0 || self.state.flip_grad_max > 50.0 {
            debug!("Vertex is singular");
            return false;
        }

        // Type 3 constraints.
        debug!("Checking Type 3 Constraints");
        if self.n > 8 {
            for (row, ref cset) in self.extra_constr.iter().enumerate() {
                for (cnum, ref c) in cset.iter().enumerate() {
                    // Constraint contains set of relevant columns.  We need to pass
                    // in the row.
                    if !c.check(&self.state.x, self.zthresh) {
                        debug!("Type 3 failure.  Row: {} Column {}", row, cnum);
                        return false;
                    }
                }
            }
        }

        // Full UY feasibility: check _bad_cols, good enforced by tableau.
        if let Some(ref uybad) = self.state.uybad {
            if uybad.iter().any(|&e| e.abs() > 1.0 + self.zthresh) {
                return false;
            }
        }

        true
    } //@}
    fn mark_visited(&mut self) { //{@
        self.set_u();
        self.set_obj();
        self.set_uy();
        self.set_grad();
        self.set_flip_grad();
        self.set_uybad();
        self.visited.insert(self.state.vertex.clone()); // Must be set in +flip+
    } //@}

    fn add_row_multiple(&mut self, tgtrow: usize, srcrow: usize, mult: f64) { //{@
        for j in 0 .. self.state.rows.ncols() {
            self.state.rows[(tgtrow,j)] += mult * self.state.rows[(srcrow,j)];//self.state.tmp[j];
        }
    } //@}

    fn div_row_float(&mut self, tgtrow: usize, divisor: f64) { //{@
        for e in self.state.rows.row_mut(tgtrow).iter_mut() {
            *e /= divisor;
        }
    } //@}
    //{@
    /// Reconstruct U from first 2 * self.n^2 entries of X.
    //@}
    fn set_u(&mut self) { //{@
        for j in 0 .. self.state.u.ncols() {
            for i in 0 .. self.state.u.nrows() {
                let idx = 2 * (self.n * i + j);
                let xplus = self.state.x[idx];
                let xminus = self.state.x[idx + 1];
                self.state.u.column_mut(j)[i] = xplus - xminus;
            }
        }

        self.state.det_u = self.state.u.determinant().abs();
    } //@}
    fn set_obj(&mut self) { //{@
        if self.state.det_u < self.zthresh {
            self.state.obj = std::f64::NEG_INFINITY;
        } else {
            self.state.obj = self.state.det_u.ln();
        }
    } //@}
    fn set_grad(&mut self) { //{@
        let u = self.state.u.clone();
        if let Some(inv) = u.try_inverse() {
            let inv = inv.transpose();
            self.state.grad.copy_from(&inv);
        } else {
            // Some sort of error here / maybe a Result?.
            self.state.grad.iter_mut().for_each(|e| *e = 0.0);
        }
    } //@}
    fn set_uy(&mut self) { //{@
        self.state.u.mul_to(&self.y, &mut self.state.uy);
    } //@}
    fn set_uybad(&mut self) { //{@
        if let Some(ref mut uybad) = self.state.uybad {
            if let Some(ref ybad) = self.ybad {
                self.state.u.mul_to(&ybad, uybad);
            }
        }
    } //@}
    fn set_flip_grad(&mut self) { //{@
        self.state.flip_grad_max = std::f64::MIN;
        self.state.flip_grad_min = std::f64::MAX;

        // Need to set flip_gradient for all self.n^2 elts of self_grad.
        for i in 0 .. self.n {
            for j in 0 .. self.n {
                let v = self.n * i + j; // index being evaluated
                let mut total_effect = 0.0;

                // Iterate over every U-var in the same row as the vertex v that
                // we are evalutating.
                for u_j in 0 .. self.n {
                    let u = self.n * i + u_j;
                    // self.urows[u] contais which +base+ row has this U-var.
                    // Here we need the coefficient sign, not X-var value sign.
                    let urow = self.urows[u].expect("ERROR: urow not set").0;
                    let sign = self.state.rows.row(urow)[2 * u];

                    // Get coefficient of the slackvar that will go 0 -> 2.
                    let coeff = self.state.rows.row(urow)[self.state.vmap[v]];

                    // Calc effect of moving this slackvar on the U-value, then
                    // multiply this delta by the gradient.
                    let delta_u = -2.0 * coeff * sign;
                    let obj_effect = delta_u * self.state.grad.row(i)[u_j];
                    total_effect += obj_effect;
                }

                // Set values.
                self.state.flip_grad[v] = total_effect;
                if total_effect < self.state.flip_grad_min {
                    self.state.flip_grad_min = total_effect;
                }
                if total_effect > self.state.flip_grad_max {
                    self.state.flip_grad_max = total_effect;
                }
            }
        }
    } //@}
    #[allow(dead_code)]
    fn print_flip_grad(&self) -> String { //{@
        let mut s = String::new();
        for (i, &e) in self.state.flip_grad.iter().enumerate() {
            s += &format!("{:-3}: {:.3}\n", i, e);
        }
        s
    } //@}

    fn is_done(&self) -> bool { //{@
        match self.state.uybad {
            Some(ref uybad) => uybad.iter().all(|&elt|
                    (elt.abs() - 1.0).abs() < self.zthresh),

            None => match self.n {
                2 => self.is_done_2(),
                3 => self.is_done_3(),
                4 => self.is_done_4(),
                5 => self.is_done_5(),
                6 => self.is_done_6(),
                7 => self.is_done_7(),
                8 => self.is_done_8(),
                _ => false,
            }
        }
    } //@}
    fn is_done_2(&self) -> bool { //{@
        self.state.det_u > self.zthresh
    } //@}
    fn is_done_3(&self) -> bool { //{@
        self.state.det_u > self.zthresh
    } //@}
    //{@
    /// All neighbors have det 8.
    //@}
    fn is_done_4(&self) -> bool { //{@
        (self.state.flip_grad_max + 0.50).abs() < 1e-2
            && (self.state.flip_grad_min + 0.50).abs() < 1e-2
    } //@}
    //{@
    /// 16, 16, 16, 16, 16,
    /// 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    /// 32, 32, 32, 32, 32, 32, 32, 32, 32, 32
    //@}
    fn is_done_5(&self) -> bool { //{@
        // Done iff a fraction of 3.
        let g = self.state.flip_grad[0] * 3.0;
        (g - g.round()).abs() < 1e-5
    } //@}
    //{@
    /// 64, 64, 64, 64, 64, 64,
    /// 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96,
    /// 128, 128, 128, 128, 128, 128, 128, 128, 128,
    /// 128, 128, 128, 128, 128, 128, 128, 128, 128
    //@}
    fn is_done_6(&self) -> bool { //{@
        
        // Done iff a fraction of 5.
        let g = self.state.flip_grad[0] * 5.0;
        (g - g.round()).abs() < 1e-5
    } //@}
    fn is_done_7(&self) -> bool { //{@
        self.state.flip_grad_max < -self.zthresh
    } //@}
    //{@
    /// All neighbors have det 3072.
    //@}
    fn is_done_8(&self) -> bool { //{@
        (self.state.flip_grad_max + 0.25).abs() < 1e-2
            && (self.state.flip_grad_min + 0.25).abs() < 1e-2
    } //@}

    pub fn dims(&self) -> (usize, usize) {
        (self.n, self.k)
    }

    pub fn has_ybad(&self) -> bool {
        self.ybad.is_some()
    }

    pub fn visited_vertices(&self) -> usize {
        self.visited.len()
    }

    pub fn get_zthresh(&self) -> f64 {
        self.zthresh
    }
} //@}
// end Tableau@}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trap_test() {
        let (n,k) = (4, 12);

        let _x = na::DMatrix::from_row_slice( n, k, 
        &vec![  1,  1,  1,  1,  1, -1,  1,  1, -1, -1,  1, -1,
                1,  1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1,
                1, -1,  1, -1,  1,  1, -1,  1,  1,  1,  1,  1,
                1, -1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1 ] );

        let y = na::DMatrix::from_row_slice( n, k,
        &vec![ 2.2025106905705982,    1.5878396886149972,  -0.7976389358074518,  3.212090908202875,
              -0.10968973241236601,  -1.5145614871755122,   3.212090908202875,  -3.212090908202875,
               1.5145614871755122,    0.10968973241236601,  1.5878396886149972, -2.2025106905705982,
               0.10784590636828145,  -3.137473232236824,    0.29856578438016174,-1.248080332640443,
              -0.7414906044239689,   -1.147902295172412,   -1.248080332640443,   1.248080332640443,
               1.147902295172412,     0.7414906044239689,  -3.137473232236824,  -0.10784590636828145,
               0.3215400214050709,   -1.8961073532383619,  -0.4164981965042077, -0.39816484167988586,          
              -0.7964503433288065,   -0.7014921682296696,  -0.39816484167988586, 0.39816484167988586,
               0.7014921682296696,    0.7964503433288065,  -1.8961073532383619, -0.3215400214050709,
               2.416491074073298,     2.3144711609857507,   4.955511945954476,   2.41651169029941,
               3.6349812453570576,   -3.7370217746707164,   2.41651169029941,   -2.41651169029941, 
               3.7370217746707164,   -3.6349812453570576,   2.3144711609857507, -2.416491074073298]);

        let u_i = na::DMatrix::from_row_slice( n, n,
        &vec![ -0.9678687665320629, -1.6886713220466487,  1.8237736996654714,  0.3010320907103666, 
                6.381111165069708,   20.72006195589735, -32.864429075374645,  -2.781640811415986,
                2.075431172633552,   8.179400876264951, -12.498693899862248, -1.0074293128335396,
                5.062623763866139,  15.494386320469278, -24.868763271438752, -2.4105971201325582 ]);


        let mut passed = false;

        let mut ft = match FlexTab::new( &u_i, &y, 1e-9 ) {
            Ok(ft) => ft,
            Err(e) => { 
                println!("Error: {}", e);
                return;
            },
        };

        match ft.solve() {
            Ok(_) => {
                println!("Solved");
                passed = true;
            },
            Err(e) => {
                println!("{}", e);
                println!("ft = {}", ft);
            },
        }

        assert!(passed);
    }

}
