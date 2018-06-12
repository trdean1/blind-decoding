function [is_permutation] = isPermute(X1,X2)
    x1sort = sortrows(X1');
    x2sort = sortrows(X2');
    x2sortneg = sortrows(-X2');
    
    if norm(x1sort - x2sort) == 0 || norm(x1sort - x2sortneg) == 1
        is_permutation = true;
    else
        is_permutation = false;
    end

    disp(num2str(min(norm(x1sort - x2sort), norm(x1sort + x2sort))));
end

