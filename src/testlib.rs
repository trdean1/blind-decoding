
extern crate nalgebra as na;
extern crate regex;

use regex::Regex;

use std;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::cmp::Ordering;
use std::thread;
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct TrialResults {
    pub dims: (usize,usize),
    pub var: f64,
    pub tol: f64,
    pub trials: usize,
    pub success: usize,
    pub not_atm: usize,
    pub complete: usize,
    pub runout: usize,
    pub error: usize,
    pub bit_errors: usize,
    pub total_bits: usize,
    pub time_elapsed: f64,
    pub badstart: usize,
    pub goodcols: usize,
    pub centerattempts: usize,
    pub linindep: usize,
    pub statestack: usize,
    pub toomanyhops: usize,
    pub trap: usize,
    pub reduced: usize,
}

impl Default for TrialResults {
    fn default() -> TrialResults {
        TrialResults {
            dims: (0,0),
            var: 0f64,
            tol: 0f64,
            trials: 0,
            success: 0,
            not_atm: 0,
            complete: 0,
            runout: 0,
            error: 0,
            bit_errors: 0,
            total_bits: 0,
            time_elapsed: 0f64,
            badstart: 0,
            goodcols: 0,
            centerattempts: 0,
            linindep: 0,
            statestack: 0,
            toomanyhops: 0,
            trap: 0,
            reduced: 0,
        }
    }
}

impl TrialResults {
    pub fn new( n: usize, k: usize, v: f64 ) -> TrialResults { 
        TrialResults {
            dims: (n, k),
            var: v,
            ..Default::default() 
        }
    }

    pub fn extended_results( &self ) {
        println!("Trials: {}, BFS reruns: {}, LinIndep: {}", 
                 self.trials, self.goodcols, self.linindep);
        println!("\tStateStack: {}, TooManyHops: {}, Trapped: {}",
                 self.statestack, self.toomanyhops, self.trap);
        println!("\tReduced: {}", self.reduced);
    }

    pub fn clear( &mut self ) {
            self.dims = (0,0);
            self.var = 0f64;
            self.tol = 0f64;
            self.trials = 0;
            self.success = 0;
            self.not_atm = 0;
            self.complete = 0;
            self.runout = 0;
            self.error = 0;
            self.bit_errors = 0;
            self.total_bits = 0;
            self.time_elapsed = 0f64;
            self.badstart = 0;
            self.goodcols = 0;
            self.centerattempts = 0;
            self.linindep = 0;
            self.statestack = 0;
            self.toomanyhops = 0;
            self.trap = 0;
            self.reduced = 0;
    }
}

impl fmt::Display for TrialResults {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();

        s += &format!( "Trials: {}, Equal ATM: {} ({:e}), Completed: {} ({:e}), Runout: {} ({:e}), Error:{} ({:e})\n",
                self.trials,self.success, self.success as f64/self.trials as f64,
                self.complete, self.complete as f64 / self.trials as f64,
                self.runout, self.runout as f64 / self.trials as f64,
                self.error, self.error as f64 / self.trials as f64
            );

        s += &format!("Bit Errors: {} / {} ({:.3e})",
                self.bit_errors, self.total_bits,
                (self.bit_errors as f64) / (self.total_bits as f64) );

        write!(f, "{}", s)
    }
}

impl std::ops::AddAssign for TrialResults {
    fn add_assign(&mut self, other: TrialResults) {
        *self = TrialResults {
            dims: self.dims,
            var: self.var,
            tol: self.tol,
            trials: self.trials + other.trials,
            success: self.success + other.success,
            not_atm: self.not_atm + other.not_atm,
            complete: self.complete + other.complete,
            runout: self.runout + other.runout,
            error: self.error + other.error,
            bit_errors: self.bit_errors + other.bit_errors,
            total_bits: self.total_bits + other.total_bits,
            time_elapsed: self.time_elapsed + other.time_elapsed,
            badstart: self.badstart + other.badstart,
            goodcols: self.goodcols + other.goodcols,
            centerattempts: self.centerattempts + other.centerattempts,
            linindep: self.linindep + other.linindep,
            statestack: self.statestack + other.statestack,
            toomanyhops: self.toomanyhops + other.toomanyhops,
            trap: self.trap + other.trap,
            reduced: self.reduced + other.reduced,

        };
    }
}

//////////////////////////////////////////////////////////////////////////////////////
///
/// This code was used to test the n=4 and n=5 claims
///
//////////////////////////////////////////////////////////////////////////////////////

// structures //{@
#[derive(Eq)]
//{@
/// Data structure for a vector of determinants.
//@}
struct Detvec {
    dets: Vec<u32>,
}

impl PartialEq for Detvec { //{@
    //{@
    /// Must both be sorted prior to calling any function that uses +eq+.
    //@}
    fn eq(&self, other: &Detvec) -> bool {
        if self.dets.len() != other.dets.len() {
            return false;
        }

        for i in 0 .. self.dets.len() {
            if self.dets[i] != other.dets[i] {
                return false;
            }
        }

        true
    }
} //@}

impl Ord for Detvec { //{@
    fn cmp(&self, other: &Detvec) -> Ordering {
        if self.dets.len() < other.dets.len() {
            return Ordering::Less;
        } else if self.dets.len() > other.dets.len() {
            return Ordering::Greater;
        }

        for i in 0 .. self.dets.len() {
            if self.dets[i] < other.dets[i] {
                return Ordering::Less;
            } else if self.dets[i] > other.dets[i] {
                return Ordering::Greater;
            }
        }

        Ordering::Equal
    }
} //@}

impl PartialOrd for Detvec { //{@
    fn partial_cmp(&self, other: &Detvec) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
} //@}

impl Detvec { //{@
    fn new(v: &Vec<u32>) -> Detvec {
        let mut dv = Detvec { dets: v.clone() };
        dv.dets.sort();
        dv
    }
} //@}

impl fmt::Display for Detvec { //{@
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = &format!("[{}]", 
                self.dets.iter().map(|e| format!("{}", e))
                .collect::<Vec<_>>()
                .join(", "));
        write!(f, "{}", s)
    }
} //@}
// end structures //@}

#[allow(dead_code)] //{@
/// For each matrix in the file +fname+, calculate the determinant pattern of
/// every neighboring matrix.  Print each unique determinant pattern.
//@}
pub fn neighbor_det_pattern(fname: &str) { //{@
    // Simple function for calculating determinant.
    let calcdet: fn(m: &na::DMatrix<f64>) -> u32 = 
        |m| m.determinant().abs().round() as u32;

    // Set number of threads to use.
    let nthreads = 4;
    // Read in all of the matrices from file.
    let maxmats = read_mtxset_file(fname);
    // Create shared data structure.
    let detvecs = Arc::new(Mutex::new(BTreeSet::new()));

    let timer = std::time::Instant::now();
    let mtx_per_thread = maxmats.len() / nthreads;
    let mut threads = Vec::with_capacity(nthreads);

    // Spawn threads.
    for tnum in 0 .. nthreads {
        let detvecs = detvecs.clone(); // clones the Arc
        let maxmats = maxmats.clone();
        let child = thread::spawn(move || {
            for i in (tnum * mtx_per_thread) .. ((tnum + 1) * mtx_per_thread) {
                println!("Thread: {}, mtx: {}", tnum, i);
                let mtx = &maxmats[i];
                let neighbor_dets = see_neighbors(&mtx, calcdet);
                let mut detvecs = detvecs.lock().unwrap();
                (*detvecs).insert(Detvec::new(&neighbor_dets));
            }
        });
        threads.push(child);
    }

    for thr in threads { thr.join().unwrap(); }
    let nanos = timer.elapsed().subsec_nanos();

    let detvecs = detvecs.lock().unwrap();
    println!("Detvecs:");
    for dv in detvecs.iter() { println!("{}", dv); }
    println!("time = {:.6}", nanos as f64 * 1e-9);
} //@}

#[allow(dead_code)] //{@
/// For each matrix in {-1, +1} ^ {n x n} for specified parameter _n_, calculate
/// the determinant of the matrix and the determinant pattern of all neighboring
/// matrices.  For each matrix determinant, print each unique neighbor pattern.
/// This is done very inefficiently: do not call with n > 5.
//@}
pub fn neighbor_det_pattern_all(n: usize) { //{@
    if n > 5 { panic!("Cannot call function with n > 5"); }

    // Foreach possible determinant value, need the set of possible detvecs.
    let mut dets = HashMap::new();

    let calcdet: fn(m: &na::DMatrix<f64>) -> u32 =
        |m| m.determinant().abs().round() as u32;

    // Generator for each \pm1 matrix of size n.
    let mut gen = gen_all_pm1(n);
    let mut num_mtx = 0;
    while let Some(mtx) = gen() {
        num_mtx += 1;
        if num_mtx & 0xf_ffff == 0 { println!("num = {:x}", num_mtx); }
        let det = calcdet(&mtx); // Get the determinant of this matrix.
        let neighbor_dets = see_neighbors(&mtx, calcdet);
        if !dets.contains_key(&det) {
            dets.insert(det, BTreeSet::new());
        }
        dets.get_mut(&det).unwrap().insert(Detvec::new(&neighbor_dets));
    }

    let mut keys = dets.keys().collect::<Vec<_>>();
    keys.sort();
    for &det in keys.iter() {
        println!("det = {}\n{}\n",
                det,
                dets.get(det).unwrap().iter()
                    .map(|ref dv| format!("{}", dv))
                    .collect::<Vec<_>>()
                    .join("\n")
                );
    }
} //@}
//{@
/// Iterate over each neighbor of +mtx+.  For each neighbor, call the +process+
/// function on that neighbor and store the result.
/// Return a vector containing the results of calling +process+ on each
/// neighbor.
//@}
pub fn see_neighbors<T>(mtx: &na::DMatrix<f64>, process: fn(&na::DMatrix<f64>) -> T) //{@
        -> Vec<T>
        where T: Ord + PartialOrd {
    let mut mtx = mtx.clone();
    let mut dets = Vec::new();
    let (nrows, ncols) = mtx.shape();

    // Iterate over each neighbor.
    for i in 0 .. nrows {
        for j in 0 .. ncols {
            // Flip to neighbor.
            let orig = mtx.row(i)[j];
            mtx.row_mut(i)[j] = if orig == -1.0 { 1.0 } else { -1.0 };

            // Call the specified _process_ function, push the result.
            dets.push(process(&mtx));

            // Flip back.
            mtx.row_mut(i)[j] = orig;
        }
    }
    dets
} //@}

#[allow(dead_code)] //{@
/// Read a set of matrices from file +fname+, return as a vector.
//@}
pub fn read_mtxset_file(fname: &str) -> Vec<na::DMatrix<f64>> { //{@
    let mut ret = Vec::new();
    let f = File::open(fname).unwrap();
    let reader = BufReader::new(f);
    let blank = Regex::new(r"^\s*$").unwrap();
    let numre = Regex::new(r"(\-?\d+)").unwrap(); 

    let data2mtx = |data: &Vec<f64>, numlines: usize| {
        if data.len() == 0 {
            return None;
        }
        if data.len() % numlines != 0 {
            println!("Input error: {} / {}", data.len(), numlines);
            return None;
        }
        Some(na::DMatrix::from_row_slice(numlines, data.len() / numlines, &data))
    };

    let mut data = Vec::new();
    let mut numlines = 0;
    for line in reader.lines().map(|l| l.unwrap()) {
        if blank.is_match(&line) {
            if let Some(mtx) = data2mtx(&data, numlines) {
                ret.push(mtx);
            }
            data = Vec::new();
            numlines = 0;
            continue;
        }

        let mut values = numre.captures_iter(&line)
            .map(|c| c[0].parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        data.append(&mut values);
        numlines += 1;
    }
    if let Some(mtx) = data2mtx(&data, numlines) {
        ret.push(mtx);
    }

    ret
} //@}
#[allow(dead_code)] //{@
/// Read a matrix from file +fname+ and return it.
//@}
pub fn read_mtx_file(fname: &str) -> na::DMatrix<f64> { //{@
    let f = File::open(fname).unwrap();
    let reader = BufReader::new(f);
    let re = Regex::new(r"(\-?\d+\.\d+)").unwrap();
    let mut data = Vec::new();
    let mut nrows = 0;
    let mut ncols: Option<usize> = None;

    for line in reader.lines().map(|l| l.unwrap()) {
        let mut values = re.captures_iter(&line)
            .map(|c| (&c[0]).parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        if values.len() == 0 { continue; }
        match ncols {
            None => ncols = Some(values.len()),
            Some(n) => assert_eq!(values.len(), n),
        };
        data.append(&mut values);
        nrows += 1;
    }

    na::DMatrix::from_row_slice(nrows, ncols.unwrap(), &data)
} //@}

#[allow(dead_code)]
//{@
/// Return closure that generates all matrices in {-1, +1} ^ {n x n}.
//@}
fn gen_all_pm1(n: usize) -> Box<FnMut() -> Option<na::DMatrix<f64>>> { //{@
    let mut count = 0;
    Box::new(move || {
        if count >= (1 << n*n) { return None; }
        // Get next matrix.
        let mut data = Vec::with_capacity(n * n);
        for i in 0 .. n*n {
            let e = if count & (1 << i) != 0 { 1.0 } else { -1.0 };
            data.push(e);
        }
        count += 1;
        Some(na::DMatrix::from_row_slice(n, n, &data))
    })
} //@}
