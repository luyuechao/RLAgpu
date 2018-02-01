% A demo on
% N. Halko Finding Structure with Randomness:
% Probabilistic Algorithms for
% Constructing Approximate Matrix Decompositions
clc; clear;close all;

addpath('./build');
fprintf ('Randomized SVD colum smapling in double-precison on CPU\n');
fprintf ('size[GB] \t m \t n \t k \t RSVD_TIME[s] \t RSVD_FRO_Err \t ERR_TIME\n');

T = 2^21;     %% starting point
m = 128;
testNumber = 20;
k = 32; %fixed rank
colData=[testNumber, 5];

for i =1:testNumber
    n = T / m;

    if k*2 > min(m,n), break; end

    A = randn(m, k) * randn(k, n);
    
    %% Randmoized SVD
     startTime = tic;
     [U, S, VT] = rSVDmex(A, 2*k, 2);
     rsvd_time = toc(startTime);
     
     %startTime = tic;
     R = A - U * diag(S) * VT;
     rsvdErr = norm(R, 'fro') / norm(A, 'fro');
     %rsvdErr = 0;
     %ErrTime = toc(startTime);
     ErrTime = 0;
    
     size = (m * n * 8) / (2^30);
    %% print out result
     fprintf('%.2e \t %d \t %d \t', size, m, n, k);
     fprintf('%.2e\t %.2e\t %.2e\t\n',... 
             rsvd_time, rsvdErr, ErrTime);
    
    colData(i, 1) = size;
    colData(i, 2) = m;
    colData(i, 3) = n;
    colData(i, 4) = k;
    colData(i, 5) = rsvd_time;
    colData(i, 6) = rsvdErr;

    m = m * 2;
    
end

%csvwrite('col.csv', colData);

fprintf ('Randomized SVD colum smapling in double-precison on CPU\n');
fprintf ('size[GB] \t m \t n \t k \t RSVD_TIME[s] \t RSVD_FRO_Err \t ERR_TIME\n');

m = 128;
rowData=[testNumber, 5];

for i =1:testNumber
    
    n = T / m;

    if k*2 > min(m,n), break; end

    A = randn(m, k) * randn(k, n);
    
    %% Randmoized SVD
     startTime = tic;
     [U, S, VT] = rSVDmex(A, 2*k, 2);
     rsvd_time = toc(startTime);
     
     %startTime = tic;
     R = A - U * diag(S) * VT;
     rsvdErr = norm(R, 'fro') / norm(A, 'fro');
     %rsvdErr = 0;
     %ErrTime = toc(startTime);
     ErrTime = 0;
    
     size = (m * n * 8) / (2^30);
    %% print out result
     fprintf('%.2e \t %d \t %d \t', size, m, n, k);
     fprintf('%.2e\t %.2e\t %.2e\t\n',... 
             rsvd_time, rsvdErr, ErrTime);
    
    rowData(i, 1) = size;
    rowData(i, 2) = m;
    rowData(i, 3) = n;
    rowData(i, 4) = k;
    rowData(i, 5) = rsvd_time;
    rowData(i, 6) = rsvdErr;

    m = m * 2;
    
end

%csvwrite('row.csv', rowData);
