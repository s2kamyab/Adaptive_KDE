function X = gmm_sample(gmm, N)
    accm = cumsum(gmm.priors);
    accm = accm / accm(end);
    
    [d M] = size(gmm.means); % d: dimension, M: number of mixtures
        
    X = zeros(d,N);
        
    gmm.vars = gmm.stds .* gmm.stds;
    
    for i = 1:N
        j = random_select(accm);
        
    
        X(:, i) = normal_gen(1, gmm.means(:, j), gmm.vars(j));
        
    end
    
    
    
end


function sample = random_select(accm)
% selects randomly a number between 1,2,...,length(accm)
% according to discrete accumulative distribution accm
    
    sample = find(rand < accm, 1);
    
    
end