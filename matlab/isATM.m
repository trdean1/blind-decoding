function [is_atm] = isATM(T)

T = round(T);
col_l1 = norms(T,1,1);
row_l1 = norms(T,1,2);

if( sum(col_l1 ~= 1) == 0 && sum(row_l1 ~= 1) == 0 )
    is_atm = true;
else
    is_atm = false;
end

end

