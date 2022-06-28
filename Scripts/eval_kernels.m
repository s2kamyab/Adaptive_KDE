function [K dK_dx Means Stds] = eval_kernels(X, KType, KOptions)
    
    switch lower(KType)
     case {'g', 'gaussian'}
      Z = KOptions{1};
      K_Stds = KOptions{2};
      
%       [K dK_dx KMeans KStds] = eval_kernels_gaussian(X, Z, K_Stds);
    [K] = eval_kernels_gaussian2(X, Z, K_Stds);
      
      
      
     case 'sinc'
      Z = KOptions{1};
      K_Stds = KOptions{2};
      
      [K KMeans KStds] = eval_kernels_sinc(X,  Z, K_Stds);
     
      if nargout > 1
          NotImplemented;
      end
     
     case {'h', 'hermite'}
      M = KOptions{1};
      mu = KOptions{2};
      sig = KOptions{3};
      
      K  = eval_kernels_hermite_poly(X,  M, mu, sig);
      
      KMeans = NaN;
      KStds  = NaN;
     
      if nargout > 1
          NotImplemented;
      end
     
      
     case {'lg', 'legendre'}
      M = KOptions{1};
      mu = KOptions{2};
      sig = KOptions{3};
      
      K  = eval_kernels_legendre_poly(X,  M, mu, sig);
      
      KMeans = NaN;
      KStds  = NaN;
     
      if nargout > 1
          NotImplemented;
      end
     
      
     otherwise
      error(['No such kernel type: ' KType]);
      
      
    end

            
end