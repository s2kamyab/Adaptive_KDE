function out = SPADES(input)
% Implementation of SPADES method as in
%Bunea F, Tsybakov AB, Wegkamp MH, Barbu A. Spades and mixture models. The Annals of Statistics. 2010 Aug;38(4):2525-58.
%the density function is parametrized as a linear combination of a number of sample points
% f_w(x) = \sum_i w_i b_i(x)
% here, each b_i(x) is a Gaussian with mean means_b(i) and standard deviation bw_b

X = input.X;
[X , ~] = sort(X);
means_b = input.means_b;
bw_b = input.bw_b;
w = input.gamma; % regularization wieght
N = length(X); % number of sample points
m_b = max(size(means_b)); % number of basis functions
p = m_b;%size(means_b,1);% number of basis functions

% introducing number of applied functions all Gaussian all with bandwidth bw and with different centres
X_min = min(X);
X_max = max(X);
% create fixed Gaussian basis functions f_j
for j = 1: m_b
    f(:,j) = normpdf(X , means_b(j) , bw_b(j));
end

%% Cooerdinate Descent Algorithm
% 1- initial all lambdas to 1/m_b
lambda = 1/m_b*ones(1,m_b);
% 2 - choose a direction j from {1...m_b} and set lambda_old = lambda

flag = 1;
while flag == 1
    j = randi(m_b);
    lambda_old = lambda;
    %3- if g_j(lambda^0_j)|<w then set lambda = lambda^0_j otherwise obtain
    %lambda by line minimization in direction j
    lambda_0_negj = lambda;
    lambda_0_negj(j)=0;
    g_j_lambda_0_negj = -(2/N)*sum(f(:,j)) + 2*f(:,j)*norm(sum(f.*repmat(lambda_0_negj , N,1) , 2));
    if abs(g_j_lambda_0_negj) < w
        lambda =  lambda_0_negj;
    else
        % use linmin to find best lambda_j
%         grid_lambda_j = 
    end
    %4 - if |lambda_old - lambda|> epsilon go to 2 epsilon = precision level
    if norm(lambda_old - lambda) <= 0.001
        flag = 0;
    end
end