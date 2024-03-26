function e = err(a)
    n = size(a,1);
    b=ones(60,1);
    b(21:60)=b(21:60)+1;
    b(41:60)=b(41:60)+1;
    e=0;
    for i=1:n
        if a(i)~=b(i)
            e = e+1;
        end
    end
    e = e/n;
end