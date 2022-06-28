%% appxml_evaluation
clear;
clc;
close all
%% initialization

density =0; % 0 -16
methods = [1,2];%1 = proposed ,2=spades%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   2 = EM , 3 = ML , 4 = approximate ML , 5 = lin nonneg , 6 = lin appx ML nonneg 7 = integrated apxml
func_methods = {@Proposed, @SPADES};
colors = ['k';'r';'g';'m'];
sparsity_vals = 0.00000001:0.000001:0.000001;
num_of_basis_functions = 600;%[200,600];
sample_num =500;% [50,100,200,400,600];
dlt = .01;
nrun =5;
min_mse = 10000000000000000;
%% Estimating pdf

th = 1e-5; % stopping threshold
dNum = 1;
% for dNum =1 : length(whichDensities)
pdf = get_pdf(density(dNum));


for i = 1 : length(sample_num)
    
    N = sample_num(i);
    input = struct();
    X = pdf_sample(pdf, N)'; % sampled data
    % find samples of y = mean(K(x))density
    X_min = min(X);
    X_max = max(X);
    m_k = 10;%N/10; % number of applied functions in proposed method
    bw_k = (X_max - X_min)/m_k;
    means_k = linspace(X_min, X_max, m_k);
    
    % compute covariance of y = 1/N \sum k(x_i) empirically
    for ysmple = 1 : 500
        X2 = pdf_sample(pdf, N)'; % sampled data
        % find samples of y = mean(K(x))
        X2_min = min(X2);
        X2_max = max(X2);
        m_k = 10;%N/10; % number of applied functions
        bw_k = (X2_max - X2_min)/m_k;
        means_k = linspace(X2_min, X2_max, m_k);
        K_X2=[];
        for gg = 1 : m_k
            K_X2(:,gg) = normpdf(X2, means_k(gg), bw_k);
        end
        %     y =
        y(:,ysmple) =mean(K_X2);
    end
    input.X = X;
    input.y = y;
    X_min = min(X);
    X_max = max(X);
    %     from = min(X)-2*std(X);
    %     to = max(X)+2*std(X);
    %     Z = from:dlt:to; % the grid
    sorted_X = sort(X);
    true_density = pdf_eval(pdf,sorted_X); % actual function
    
    input.th = th;
    plot(sorted_X,true_density,'LineWidth' , 2);grid on;
    %% % define the basis functions. A number of gaussians with bandwidth bw_b and centres listed in means_b
    for M =1% 1 : length(num_of_basis_functions)
        q = 10;
        input.m_b = num_of_basis_functions(M);% number of basis functions
        for c = 1 : q
            for d = 1 : num_of_basis_functions(M)/q
                bw_b(c,d) = c *20*(X_max - X_min)/input.m_b;
            end
        end
        input.bw_b = bw_b(:);
        input.means_b =linspace(X_min+bw_b(end), X_max-bw_b(end), input.m_b);% +-largest bandwidth X2';%linspace(X_min+bw_b(10), X_max-bw_b(10), input.m_b);% +-largest bandwidth
        %%
        for g =1 : length(sparsity_vals)
            input.gamma = sparsity_vals(g);
            for r = 1 : nrun
                densityNum = density(dNum)
                runNum = r
                for n = 1 : length(methods)
                    m = methods(n);
                    method = func_methods{m}
%                     input = spades_tuner(input,m);
                    tic
                    out = func_methods{m}(input);
                    time_me(i, r , n) = toc;
                    w = out.w;
                    B_Z = eval_kernels(sorted_X, 'Gaussian', {input.means_b, input.bw_b});
%                     B_X = eval_kernels(X, 'Gaussian', {input.means_b, input.bw_b});
                    
                    estimated_density = w'*B_Z;
                    MSE(g, r, n) = immse(true_density , estimated_density');%sum(sum((true_density - estimated_density).^2))/N;
                    if MSE(g, r, n) < min_mse
                    hold on;plot(sorted_X,estimated_density,colors(n),'LineWidth' , 2);
                    end
                    
                end
                
%                                 legend('Original density' , 'Proposed AKDE' , 'SPADES');
                disp('====================================================');
                fprintf('               END OF RUN %d           \n' , r);
                disp('====================================================');
                
                
            end
        end
    end
end
MSE_proposed = mean(MSE(:,:,1), 2);
MSE_spades = mean(MSE(:,:,2), 2);
% figure;plot([50,100,200,400,600], [MSE_proposed';MSE_spades'], 'LineWidth' , 2);
figure;plot(sparsity_vals, [MSE_proposed';MSE_spades'], 'LineWidth' , 2);
legend( 'Proposed AKDE' , 'SPADES');
xlabel('\gamma');ylabel('Average MSE');
grid on;