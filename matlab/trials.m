%Crude code to demonstrate the original MATLAB implementation.
%Ask me if you want more test framework...I have a lot but it's 
%not organized

numTrials = 10;
n=4;
s = zeros(0,15);
for k = 1:15
    p = 0;
    for i = 1:numTrials
        [u,h] = trialTransform(n,2,n+k);
        if abs(det(u*h))-1.0 < 0.01 % For n=4 this implies recover up to an ATM
            p = p+1;
        end
    end
    s(k+1) = p / numTrials;
end

s