isATM (function):	purpose: to check if input matrix in an ATM after rounding
			input: matrix
			output: boolean

test_single_run (script): this was an initial testing script to which was then generalized to test_perm_loop. The algorithm is described explicitly in the report, so we will only give a quick outline here. We randomly generate a gain matrix H, define an input X to ensure  recoverability (eg full rank and few other conditions specified in Dean et al), add noise (if noiseLevel != 0), and attempt to revoer H_perm for a variety of permutations of Y, called Y_perm. The resultant "recovered" H_perm's, called A_perm in the code, are then averaged to a  final estimate A. The error of this estimate, measured as the frobenius norm of the difference from the ground-truth H, was then compared with the average errors of all the A_perms.

test_perm_loop (script): this uses test_single_run as a basis and simply loops for a single randomly generated gain matrix H and noise, varying the number of permutations used. It plots the errors discussed for test_single_run against the number of permutations used, as well as the difference between these errors. NOTE 1: This code is inefficient and recomputes the results for permutations, but since it was just a proof of concept and the runtime wasn't horrific, we did not worry about this. NOTE 2: As discussed in the report and paper, the given in those two documents is the result of running this script multiple times for different rng seeds and averaging the results. 

newton_step_test.m (script): this script tests our initialization scheme for multiple sets of received symbols, each with their own true channel matrix and transmitted symbols.  Plots minimum distance over iteration for all trials that are primal feasible.  Calls newton_step_trial.m.

newton_step_trial.m (function):  purpose:	to implement our initialization scheme for a single set of received symbols.  As we worked with BPSK in the noiseless case, our goal is to minimize -log|det U| subject to norm(U*Y,inf) <= 1.  While the problem is non-convex, solutions occur on the vertices of the feasible region, i.e. U*Y is in {+1,-1}^(nxk).  Thus, we reformulate the problem, add a log barrier, and use an infeasible start Newton method with the goal of getting U*Y close to a vertex.  A more detailed formulation is found in our report. Returns a starting value for U. 

numerical_gradient_test.m (script):  We used this script to ensure that the gradients and hessians we had derived to solve the problem with the Newton method were indeed correct.

