%% appxml_evaluation
clear;
clc;
close all
%% initialization

density_num =4; % 0 -16
methods = [1,2];%1 = proposed ,2=spades
func_methods = {@Proposed, @SPADES};
colors = ['k';'r';'g';'m'];
sparsity_vals = 20:10:400;
num_of_basis_functions = 100;%[200,600];
sample_num =1000;% [50,100,200,400,600];
dlt = .01;
nrun =50;
min_mse_proposed = 10000000000000000;
min_mse_spades = 10000000000000000;
%% Estimating pdf

% pdf = get_pdf(density_num);


for i = 1 : length(sample_num)
    
    N = sample_num(i);
    input = struct();
    
    X = generateSamples(density_num , N , 1);
    % find samples of y = mean(K(x))density
    X_min = min(X);
    X_max = max(X);
    m_k = 10;%N/10; % number of applied functions in proposed method
    bw_k = (X_max - X_min)/m_k;
    means_k = linspace(X_min, X_max, m_k);
    
    % compute covariance of y = 1/N \sum k(x_i) empirically
    for ysmple = 1 : 50
        X2 = generateSamples(density_num , N , 1);%pdf_sample(pdf, N)'; % sampled data
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
        y(:,ysmple) =mean(K_X2);
    end
    input.X = X;
    input.y = y;
    X_min = min(X);
    X_max = max(X);
    sorted_X = sort(X);
    true_density = computeDensities(X , density_num);%pdf_eval(pdf,sorted_X); % actual function
    plotDensities(sorted_X , density_num)
    %% % define the basis functions. A number of gaussians with bandwidth bw_b and centres listed in means_b
    for m =1% 1 : length(num_of_basis_functions)
        q = 10;
        input.m_b = num_of_basis_functions(m);% number of basis functions
        for c = 1 : q
            for d = 1 : num_of_basis_functions(m)/q
                bw_b(c,d) = c *(X_max - X_min)/input.m_b;
            end
        end
        input.bw_b = bw_b(:);
        input.means_b =linspace(X_min+bw_b(end), X_max-bw_b(end), input.m_b);% +-largest bandwidth X2';%linspace(X_min+bw_b(10), X_max-bw_b(10), input.m_b);% +-largest bandwidth
        %%
        for g =1 : length(sparsity_vals)
            input.gamma = sparsity_vals(g);
            for r = 1 : nrun
                densityNum = density_num
                runNum = r
                for n = 1 : length(methods)
                    mm = methods(n);
                    method = func_methods{mm}
                    tic
                    out = func_methods{mm}(input);
                    time_me(i, r , n) = toc;
                    w = out.w;
                    for t = 1 : input.m_b % fixed basis functions
                        f(:,t) = normpdf(sorted_X, input.means_b(t), input.bw_b(t));
                    end
                    estimated_density = w*f';
                    MSE(g, r, n) = immse(true_density , estimated_density');%sum(sum((true_density - estimated_density).^2))/N;
                    if MSE(g, r, n) < min_mse_proposed & n == 1
                        estimated_density_proposed = estimated_density;
                        min_mse_proposed = MSE(g, r, n);
                    end
                    if MSE(g, r, n) < min_mse_spades & n == 2
                        estimated_density_spades = estimated_density;
                        min_mse_spades = MSE(g, r, n);
                    end
                    
                end
                
                disp('====================================================');
                fprintf('               END OF RUN %d           \n' , r);
                disp('====================================================');
                
                
            end
        end
    end
end
MSE_proposed = mean(MSE(:,:,1), 2);
MSE_spades = mean(MSE(:,:,2), 2);

hold on;plot(sorted_X,estimated_density_proposed,colors(1),'LineWidth' , 2);
hold on;plot(sorted_X,estimated_density_spades,colors(2),'LineWidth' , 2);
legend('GT','Proposed AKDE', 'SPADES'); grid on;

figure; 
subplot(1,2,1);plot(sparsity_vals, MSE_proposed', 'LineWidth' , 2);legend( 'Proposed AKDE' ); grid on;xlabel('\gamma');ylabel('Average MSE');
subplot(1,2,2); plot(sparsity_vals, MSE_spades' , 'LineWidth' , 2);legend( 'SPADES');xlabel('\gamma');ylabel('Average MSE');grid on;