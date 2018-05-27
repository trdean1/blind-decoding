%Crude code to demonstrate the original MATLAB implementation.
%Ask me if you want more test framework...I have a lot but it's 
%not organized

numTrials = 10;
n=4;
kmax = 10; %15;
kmin = 10;
s = zeros(0,15);
for k = kmin:kmax
    p = 0;
    for i = 1:numTrials
        [u,h] = trialTransform_logbarrier(n,2,k);
        if abs(det(u*h))-1.0 < 0.01 % For n=4 this implies recover up to an ATM
            p = p+1;
        end
    end
    s(k+1) = p / numTrials;
end

s(k+1)