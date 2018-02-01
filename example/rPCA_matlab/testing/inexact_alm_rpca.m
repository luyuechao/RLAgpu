function [LO, SP, Y, mu, iter] = inexact_alm_rpca(M, lambda, tol, maxIter)

% Oct 2009
% This matlab code implements the inexact augmented Lagrange multiplier 
% method for Robust PCA.
%
% M - m x n matrix of observations/data (required input)
%
% lambda - weight on sparse error term in the cost function
%
% tol - tolerance for stopping criterion.
%     - DEFAULT 1e-7 if omitted or -1.
%
% maxIter - maximum number of iterations
%         - DEFAULT 1000, if omitted or -1.
% 
% Initialize A,E,Y,u
% while ~converged 
%   minimize (inexactly, update A and E only once)
%     L(A,E,Y,u) = |A|_* + lambda * |E|_1 + <Y,M-A-E> + mu/2 * |M-A-E|_F^2;
%   Y = Y + \mu * (D - A - E);
%   \mu = \rho * \mu;
% end
%
% Minming Chen, October 2009. Questions? v-minmch@microsoft.com ; 
% Arvind Ganesh (abalasu2@illinois.edu)
%
% Copyright: Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%            Microsoft Research Asia, Beijing

[m,n] = size(M);

if(nargin < 2), lambda = 1 / sqrt(m); end
if(nargin < 3), tol = 1e-7;     elseif(tol == -1), tol = 1e-7; end
if(nargin < 4), maxIter = 50; elseif(maxIter == -1), maxIter = 1000; end

% initialize
%norm_two = norm(M, 2);  %% get the largest singular value 
norm_inf = norm(M(:), inf) / lambda;

%Y = M / max(norm_two, norm_inf);
Y = M / norm_inf;

LO = zeros(m, n);
SP = zeros(m, n);
%mu = 1.25 / norm_two; % this one can be tuned
mu = 1.25 / norm_inf; % this one can be tuned
mu_limit = mu * 1e7;
rho = 1.5;            % this one can be tuned
M_fro_norm = norm(M, 'fro');

% fprintf('lambda = %0.3e \t |M|_inf = %0.3e \t mu = %0.3e\n',lambda, norm_inf, mu);
sv = 10;
temp_T = M - LO + (1/mu) * Y;
SP = max(temp_T - lambda/mu, 0);
SP = SP + min(temp_T + lambda/mu, 0);
    
for iter = 1:maxIter
    
    [U, S, V] = rsvd(M - SP + (1/mu) * Y, 20);
    diagS = diag(S);
    
    %% shrinkage
    svl = length(find(diagS > 1/mu));%% singular value length

    if svl < sv
        sv = min(svl + 1, n);
    else
        sv = min(svl + round(0.05*n), n);
    end
    
    LO = U(:, 1:svl) * diag(diagS(1:svl) - 1/mu) * V(:, 1:svl)';
    mu = min(mu * rho, mu_limit);
 
    Z = M - LO - SP;
    Y = Y + mu * Z;   
    temp_T = M - LO + (1/mu) * Y;
    SP = max(temp_T - lambda/mu, 0);
    SP = SP + min(temp_T + lambda/mu, 0);    
    %% stop Criterion
    errZ = norm(Z, 'fro') / M_fro_norm; 
    if errZ < tol
        break;
    end
    
    %printf('%d \t %0.3e \t\t %d \n', iter, errZ , svl);
    %disp(['#iter ' num2str(iter) ' r(A) ' num2str(rank(LO))...
    %     ' |E|_0 ' num2str(length(find(abs(SP)>0)))...
    %    ' stopCriterion ' num2str(stopCriterion)]);
    

end
if(iter == maxIter), disp('Maximum iterations reached'); end

end

% randmoized SVD
function [U,S,V] = rsvd(A,K)
p = 1; %% power iteration
[m, n] = size(A);

if m > n %% column sampling
    %% stage A:
    l = min(2 * K, n);   % l = k + p, p is set to k here
    Omega = randn(n, l); % Omega: nxl
    Y = A * Omega;       % Y: mxl
    % power iteration
    for i = 1:p
    P = A' * Y;
    Y = A * P;         %% power iteration
    end

    %% fprintf('size of Y = %d \n', size(Y));
    [Q, ~] = qr(Y, 0);         % Q: mxl
    %% fprintf('size of Q = %d \n', size(Q));
    %% stage B:
    B = Q' * A;          % B: lxn
    %% fprintf('size of B = %d \n', size(B));

    [U_tilde, S, V] = svd(B, 'econ'); % U_tilde: lxl

    U = Q * U_tilde;     % U: mxl

else %% row sampling

    l = min(2 * K, m);   % l = k + p, p is set to k here
    Omega = randn(l, m); % Omega: lxm
    Y = Omega * A;       % Y: lxn
    % power iteration
    for i = 1:p
    P =  Y * A';      %%P: l*m
    Y = P * A;         %% Y: lxn
    end
    Q = qr(Y', 0);         % Q: nxl

    %% stage B:
    B = A * Q;          % B: mxl
    %fprintf('size of B = %d \n', size(B));
    [U, S, V_hat] = svd(B, 'econ');
    V = Q * V_hat;     % V: nxl

end %% end of if m>n

end


