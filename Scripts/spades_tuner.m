function out = spades_tuner(input, samples, M, method)
indices = crossvalind('Kfold',samples,10);
n = length(input.X);
for i = 1:10
    test = (indices == i);
    train = ~test;
    for k = 1 : M
        out = Generalized_Bisection_Method(train, input, method);
        gamma = out.gamma;
        I = out.I;
        L(i,k) = get_unpenalized_objective_value(test , input, gamma, I, method);
        
    end
    
    
end
for k = 1 : M
    LL(k) = (1/size(L,1)) *sum(L(:,k));
    tt(k) = LL(k) + 0.5*k* log(n)/n;
end
[~,k_opt]=min(tt);


