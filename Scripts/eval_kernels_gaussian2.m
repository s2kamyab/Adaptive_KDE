function [K] = eval_kernels_gaussian2(X, means, stds)
  nStds = length(stds);
  nMeans  = length(means);
  
  %% by S. Kamyab
  for i = 1 : nMeans
      K(i , :) = normpdf(X , means(i) , stds(i))';
  end
  %%
  
  
%   L = nStds * nMeans;
%    
%   [d N] = size(X);
%   
%   K = zeros(L,N);
%   for i = 1:nStds
%     s = stds(i);
%     
%     ind_range = (i-1)*nMeans+1:i*nMeans;
%     
%     X2 = repmat(X, [nMeans 1]);
%     Mu2 = repmat(means', [1 N]);
%         
%     K(ind_range, :) = normpdf(X2,Mu2, s);
%     
%   end
%   
%   
%   
%   if nargout < 2; return; end
%   
%   dK_dx = zeros(L,N);
%   for i = 1:nStds
%     s = stds(i);
%     
%     ind_range = (i-1)*nMeans+1:i*nMeans;
%     
%     X2 = repmat(X, [nMeans 1]);
%     Mu2 = repmat(means', [1 N]);
%         
%     dK_dx(ind_range, :) = (Mu2-X2) .* K(ind_range, :) /s/s;
%     
%     
%   end
%   
%   
%   
%   if nargout < 3; return; end
%   
%   Means = zeros(L,1);
%   Stds = zeros(L,1);
%   
%   for i = 1:nStds
%     s = stds(i);
%     
%     ind_range = (i-1)*nMeans+1:i*nMeans;
%     
%     
%     Means(ind_range) = means;
%     Stds(ind_range) = s;      
%     
%   end
%   
  
  
  
  
  
% $$$   
% $$$   ind = 1;
% $$$   tic
% $$$   for s = stds
% $$$     for mu = means
% $$$       
% $$$       K(ind, :) = normpdf(X,mu, s);
% $$$            
% $$$       dK_dx(ind, :) = (mu-X) .* K(ind, :) /s/s;
% $$$       
% $$$       Means(ind) = mu;
% $$$       Stds(ind) = s;
% $$$       
% $$$       ind = ind + 1;
% $$$     end
% $$$   end
% $$$   toc
% $$$   
% $$$   K2 = K;
% $$$   dK_dx2 = dK_dx;
% $$$   Means2 = Means;
% $$$   Stds2 = Stds;
% $$$   
% $$$   
% $$$    
% $$$   K = zeros(L,N);
% $$$   Means = zeros(L,1);
% $$$   Stds = zeros(L,1);
% $$$   
% $$$   clear dK_dx
% $$$   
% $$$   
% $$$   
% $$$   for i = 1:nStds
% $$$     s = stds(i);
% $$$     
% $$$     ind_range = (i-1)*nMeans+1:i*nMeans;
% $$$     
% $$$     X2 = repmat(X, [nMeans 1]);
% $$$     Mu2 = repmat(means', [1 N]);
% $$$         
% $$$     K(ind_range, :) = normpdf(X2,Mu2, s);
% $$$     dK_dx(ind_range, :) = (Mu2-X2) .* K(ind_range, :) /s/s;
% $$$     
% $$$     Means(ind_range) = means;
% $$$     Stds(ind_range) = s;      
% $$$     
% $$$   end
% $$$   
% $$$   
% $$$   
% $$$   dd = max(max(abs(K2-K)))
% $$$   dk = max(max(abs(dK_dx2-dK_dx)))
% $$$   dm = max(abs(Means2-Means))
% $$$   ds = max(abs(Stds2-Stds))
% $$$   
% $$$   
% $$$   
% $$$   
% $$$   
% $$$   
% $$$   exitNOW
  
  
end