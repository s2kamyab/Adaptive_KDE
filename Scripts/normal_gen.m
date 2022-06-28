function out = normal_gen( n, mean, cov_mtx )
    % make and returns n d-dimentinal points with normal distribution
    % with mean 'mean' and covariance matrix 'cov_mtx'
    % each point is a column of the output
    
    d = length(mean);
    
    [V,E] = eig(cov_mtx);
    Aw = E^(-.5)*V;
    
    Awi = inv(Aw);

    out = zeros(d,n);
    for i = 1:n
        out(:,i) = Awi' * randn(d,1) + mean;   
    end
    
    
    
end
