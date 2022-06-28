function out = SPADES(input)
% the density function is parametrized as a linear combination of a number of sample points
% f_w(x) = \sum_i w_i b_i(x)
% here, each b_i(x) is a Gaussian with mean means_b(i) and standard deviation bw_b
% this function calculates the maximum likelihood using the approximate method
X = input.X;
[X , ~] = sort(X);
means_b = input.means_b; 
bw_b = input.bw_b;
gamma = input.gamma;
N = length(X); % number of sample points
m_b = max(size(means_b)); % number of basis functions
p = m_b;%size(means_b,1);% number of basis functions

% introducing number of applied functions all Gaussian all with bandwidth bw and with different centres
X_min = min(X);
X_max = max(X);

for i = 1% : m_k
    for j = 1: m_b
        b(:,j) = normpdf(X , means_b(j) , bw_b(j,i));
    end
end
for i = 1 : m_b
    for j = 1 : m_b
        H(i,j) = trapz(b(:,i) , b(:,j));
    end
end


%% generating A in Aw<=b , b = zeros(2*m_b,1)
A = zeros(p);
% for i = 1 : p
%     A(i , i) = -1;
%     A(i , m_b+i) = 1;
%     A(2*m_b-i+1 , 2*m_b-i+1)=1;
%     A(2*m_b-i+1 , m_b-i+1)=1;
% end
for i = 1 : p
    A(i , i) = -1;
    A(i , m_b+i) = -1;
%     A(2*m_b-i+1 , 2*m_b-i+1)=-1;
    A(m_b+i , i)=-1;
%     A(2*m_b-i+1 , m_b-i+1)=1;
    A(m_b+i , m_b+i)=1;
end
H_quadprog = [H, zeros(m_b, m_b) ; zeros(m_b , 2*m_b)];
f = [-2/p*sum(b),  gamma * ones(1 , m_b)];



%w = (KSi*KB)\(KSi*kb);
%   w = [KSi*KB ones(m_b,1); ones(1,m_b) 0]\[KSi*kb; 1];
out = struct();
w = quadprog(H_quadprog, f, A , zeros(2*m_b , 1));
w = w(1:m_b)';
w = w / sum(w);
out.w = w';
%w2 = lscov(KB,kb,Sig);






end