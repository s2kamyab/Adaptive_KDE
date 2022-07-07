function out = Proposed(input)%(X, means_b, bw_b , gamma)
% the density function is parametrized as a linear combination of a number of sample points
% f_w(x) = \sum_i w_i b_i(x)
% here, each b_i(x) is a Gaussian with mean means_b(i) and standard deviation bw_b
% this function calculates the maximum likelihood using the approximate method
X = input.X;
[X , ~] = sort(X);
means_b = input.means_b; 
bw_b = input.bw_b*50;
gamma = input.gamma;
N = length(X); % number of sample points
m_b = max(size(means_b)); % number of basis functions
p = m_b;%size(means_b,1);% number of basis functions

% introducing number of applied functions all Gaussian all with bandwidth bw and with different centres
X_min = min(X);
X_max = max(X);
m_k = 10;%N/10; % number of applied functions
bw_k = (X_max - X_min)/m_k;
means_k = linspace(X_min, X_max, m_k);
for i = 1 : m_k
    K_X(: , i) = normpdf(X, means_k(i), bw_k);
end
y = mean(K_X)';



Sig = cov(input.y');
% Sig = est_Sig_Y_data(K_X');
% Sig = est_Sig_Y_data(K_X');
%Sig = est_Sig_Y_init(X, 'Gaussian', {means, bw}, 2);


%   KB = normpdf(repmat(means', 1, m_b), repmat(means_b, m, 1),sqrt(bw*bw+bw_b*bw_b));
% KB = normpdf(repmat(means', 1, m_b), repmat(means_b, m, 1),sqrt(bw*bw+bw_b.*bw_b));
for i = 1% : m_k
    for j = 1: m_b
        b(:,j) = normpdf(X , means_b(j) , bw_b(j,i));
    end
end
for i = 1 : m_k
    for j = 1 : m_b
        H(i,j) = trapz(X , K_X(:,i).*b(:,j));
    end
end

Sig_inv = inv(Sig);
% KSi = KB'*Sig_inv;

%% generating A in Aw<=b , b = zeros(2*m_b,1)
A = zeros(p);
for i = 1 : p
    A(i , i) = -1;
    A(i , m_b+i) = -1;
%     A(2*m_b-i+1 , 2*m_b-i+1)=-1;
    A(m_b+i , i)=-1;
%     A(2*m_b-i+1 , m_b-i+1)=1;
    A(m_b+i , m_b+i)=1;
end
% Aeq = zeros(1,2*p);
% Aeq(1 , 1:m_b) = ones(1,m_b);
% tt = Sig\H; % instead of Sig_inv*H
% L = zeros(m_b);
% s = 2;
% for i = 1 : m_b
%     for j = 1 : m_b
%         if abs(i-j) <= s/2
%         L(i , j) = 1;
%         else
%          L(i , j) = 0;   
%         end
%     end
% end
% beta = 0.5;
% H_quadprog = [2 * H' *Sig_inv*H - beta *L, zeros(m_b, m_b) ; zeros(m_b , 2*m_b)];
H_quadprog = [H' *Sig_inv*H, zeros(m_b, m_b) ; zeros(m_b , 2*m_b)];
% f = [2 * y'*tt,  gamma * ones(1 , m_b)];
f = [-2 * (H'*Sig_inv) *y;  gamma * ones(m_b,1)];



%w = (KSi*KB)\(KSi*kb);
%   w = [KSi*KB ones(m_b,1); ones(1,m_b) 0]\[KSi*kb; 1];
out = struct();
w = quadprog(H_quadprog, f, A , zeros(2*m_b , 1));%, Aeq, 1);
w = w(1:m_b)';
% w = w / sum(w);
out.w = w;
%w2 = lscov(KB,kb,Sig);






end