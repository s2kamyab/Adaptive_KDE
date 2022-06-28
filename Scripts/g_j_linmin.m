function out = g_j_linmin()
global f lambda k N
out = -(2/N)*sum(f(:,k)) + 2*f(:,k)*norm(sum(f.*repmat(lambda , N,1) , 2));
