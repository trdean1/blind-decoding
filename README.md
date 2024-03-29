# blind-decoding

#### src
Contains the main rust source code.  JP is in the process of refactoring this code and we have a number of optimizations yet to be implmented. Feel free to modify and reorganize as you see fit.

You may need to install the nalgebra crate to get this to compile (should be as simple as 'cargo install nalgebra')

#### python
This contains my implementation of the "dynamic" step (Algorithm 1 in the Asilomar paper).  This does not implement the block optimization discussed in the references.  JP has the full solver implemented in Python.  Ask if you want a copy.

#### matlab
Code from the original paper submitted to IT Trans. This just uses fmincon to solve the optimization problem.  The code included here is fairly simple and just tests for ATM recovery at n=4.  I have a lot of different test frameworks that I build for MATLAB that I can provide but all my MATLAB code is highly disorganized.

#### ref
Some potentially useful references.  Ask if you want the source for any of these when writing up your project

* __IT_Trans_paper.pdf__: Our paper submitted to Trans. on Info Theory which documents the interior-point based algorithm (this is on my website and on the arXiv).
* __asilomar_final.pdf__: Our conference paper which tersely describes the fast solver
* __awgn.pdf__: These are my notes on how to adopt the solver to handling AWGN.  The AWGN code is not complicated but entirely driven by heuristics.  
* __dynamic_overview.pdf__: A slightly more in-depth description of Algorithm 1 from the Asilomar paper.
* __projection_method.pdf__: This describes how we compute lines 4 and 5 of Algorithm 1 from the Asilomar paper.  You can also perform this calculation by just calling SVD but this is very slow.  This method is currently implemented but we do not take into account the sparse, block structure of the matrix p.  We should be able to shave off at least a factor of n from the current implementation
* __JP_summer_report.pdf__: JP's write up of his work over the summer.  This is
  a little less relivant to you project as it was mostly about his
implementation of a vanilla simplex solver as well as some more theoretical work
on the n=5 case.
* __JP_fall_report.pdf__: JP's writeup from the fall quarter.  This contains a
  pretty detailed overview of the internals of the simplex portion of our solver
* __JP_winter_report.pdf__: JP's final report.  Currently the most comprehensive
  documentation of the solver.
