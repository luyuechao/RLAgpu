function demo_iALM
% Demo for iALM
%
clear;close all;

addpath('../build');

%% Change settings here to est different problems
%==================================================================
spr = 0.05;    % sparsity ratio: #nonzeros/m/n
rB = 20;     % rank of Low-Rank matrix
%==================================================================
%% generate problem
% problem size
m = 23040;
n = 1700;

fprintf('m \t n \t k \t sparsity  Iter\t time[s] \t |L-L_gt|_f \t |S-S_gt|_f \t \n');
    
% Low-Rank matrix
LO = randn(m, rB) * randn(rB, n);

% Sparse matrix
SP = zeros(m, n);
p = randperm(m * n);
sp_size = round(spr * m * n); %% the number of total sparse point
SP(p(1:sp_size)) = randn(sp_size, 1);

%  Low-Rank + Sparse
M = LO + SP;

%% main process
startTime = tic;
[LowRank, Sparse, iter] = rPCAmex(M, 1 / sqrt(m), 1e-7, 100, rB);

%[LowRank, Sparse, ~, ~, iter] = inexact_alm_rpca(M);
elapsedTime = toc(startTime);
LOerr = norm(LO - LowRank, 'fro') / norm(LO, 'fro');
SPerr = norm(SP - Sparse,  'fro') / norm(SP, 'fro');

fprintf('%d \t %d \t %d \t %0.3f \t  %d \t %0.3e \t %0.3e \t %0.3e \n',... 
         m,     n,   rB,     spr,    iter, elapsedTime, LOerr, SPerr);
    
tx = 13;
figure;  set(1,'position',[0,100,1000,1000]);
subplot(221); imshow(SP,[]);      title('True Sparse','fontsize',tx); axis square;
subplot(222); imshow(Sparse,[]);  title('Recovered Sparse','fontsize',tx); axis square;
colormap(bone(5));

subplot(223); imshow(LO,[]);      title('True Low-Rank','fontsize',tx); axis square;
subplot(224); imshow(LowRank,[]); title('Recovered Low-Rank','fontsize',tx); axis square;
colormap(bone(5));

end
