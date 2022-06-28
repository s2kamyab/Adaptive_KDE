function P = gmm_eval(gmm, Z)
    [d M] = size(Z); % d: dimention, M: number of points
    
    P = zeros(1,M);
    
    for i = 1:size(gmm.means,2)
        P = P + gmm.priors(i) * normpdf(Z, gmm.means(:,i), gmm.stds(i));
    end
    
    
    
end