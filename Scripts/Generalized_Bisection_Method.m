function out = Generalized_Bisection_Method(samples, input, M, method)
% Generalized bisection method from SPADES and Mixture models by Florentina
% Bunea et. al.

% Initialize all wi with -1.
gamma = -1*ones(1,M);
% 1. Choose w0 very large, such that bn(w0) = 0. Choose wn = 0, hence, bn(wn) = n.
gamma(1) = 10000;
gamma(end) = 0;

% 2. Initialize a queue q with the pair (w0,wn).
q = [gamma(1),gamma(end)];
% 3. Pop the first pair (a, b) from the queue.
while size(q,1)>0
    tmp=q(1,:);
    q(1,:)=[];
    % 4. Take w = (a + b)/2. Compute k = bn(w).
    ww = (tmp(1) + temp(2))/2;
    k = n_hat(samples, input, ww, method);
    % 5. If wk = -1, make wk = w.
    if gamma(k) == -1
        gamma(k) = ww;
    end
    n_a = n_hat(input, tmp(1), method);
    % 6. If |n(a)  - k| > 1 and |a - w| > alpha, add (a,w) to the back of the queue.
    if (abs(n_a - k) > 1 & abs(tmp(1) - ww) > alpha)
        q(end+1 , :) = [tmp(1), ww];
    end
    % 7. If |n(b) - k| > 1 and |b - w| > alpha, add (w, b) to the back of the queue.
    n_b = n_hat(input, tmp(2), method);
    if (abs(n_b - k) > 1 & abs(tmp(2) - ww) > alpha)
        q(end+1 , :) = [ww, tmp(2)];
    end
    % 8. If the queue is not empty, go to 3.
end
out = struct();
out.gamma = gamma;
out.I = find(gamma>0.01);