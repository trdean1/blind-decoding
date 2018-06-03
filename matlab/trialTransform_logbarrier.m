function [ U,H ] = trialTransform( n, pamSize, k, num_steps)
    %Generates a random n-by-n channel gain matrix and tests
    %recovery with a block of X symbols that are n-by-k and 
    %drawn uniformly from the constellation of size pamSize

    %Random square channel
    H = randn(n);
    
    %Make symbols
    X = 2*randi(pamSize,n,k) - pamSize - 1;

    %Test with no noise 
    Y = H*X;
    
    %x0 is a random unitary matrix
    rm = randn(n);
    [q,r] = qr(rm);
    u0 = 0.1*reshape(q,[n^2,1]);
    
    
    % Take several centering steps
    ut = u0;
    alpha = 0.2;
    beta = 0.2;
    M = pamSize;
    ut = newton_cent(Y, M, ut, num_steps, alpha, beta);
    
    %cA and cB are the contraints passed to fmincon
    cA = construct_constraints(transpose(Y), n, k);
    cB = pamSize*ones(2*n*k,1)-1;
    obj = @det_from_list;
    options = optimoptions('fmincon','MaxFunctionEvaluations',25000,'OptimalityTolerance',1e-8, 'StepTolerance',1e-7);
    U_flat = fmincon(obj,ut,cA,cB,[],[],[],[],[],options);
    U = transpose(reshape(U_flat,[n,n]));
end

function [ A ] = construct_constraints( y, n, k )
    %Uses y to construct a series of contraints corresponding
    %to constraining the l-infinity norm of U*Y
    s = size(y);
    n = s(2);
    samples = s(1);
    A = zeros(2*samples,2*n);
    for i = 1:samples
        for j = 1:n
            for k = 1:n
                A(2*n*(i-1) + j, (j-1)*n+k)  = y(i,k);
                A(2*n*(i-1) + n + j,(j-1)*n+k ) = -1*y(i,k);
            end
        end
    end
end

function [ f,gradf ] = det_from_list( x )
    %Return the objective function and gradient
    %there is some scaling here to make fmincon work better
    s = size(x);
    n = sqrt(s(1));
    m = zeros(n,n);
    m=reshape(x,[n,n]);
    f = -5000000*log(abs(det(m)));
    gradf = reshape(-1*5000000*inv(m)',[n^2,1]);
end

