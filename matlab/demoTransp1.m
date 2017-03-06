%%------------ Illustrates difference between row and column access

n=40;
%% Create a random sparse matrix
A = sprand(n, n, 0.05) + speye(n);
B = kron(A,A);
T = sptridiag(1, -2, 1, 10);
C = kron(T,B);
%% spy(C);
C = C*C;
N = size(C,1);
dens = nnz(C)/N

%% Random vector
v = randn(N,1);
t = 0;

%% Time access by rows
tic
for it = 1:10
    for i=1:n
        [ii jj vv] = find(C(i,:));
        t = t+vv*vv';    %% vv is a row
    end
end
toc

t = 0;
%% Time access by columns
for it = 1:10
    for j = 1:n
        [ii jj vv] = find(C(:,j));
        t = t+vv'*vv;    %% vv is a column
    end
end
toc
