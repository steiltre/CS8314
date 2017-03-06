function [A] = assembl(tri, XY)
%% Assembles the adjacency matrix from triangles.
%% Constant==1 in each triangle.
%% XY used only to get dimensions
[n,m] = size(XY);

%% Make adjacency matrix
A = sparse(n,n);
for i = 1:size(tri,1)
    K = tri(i,:);
    Ae = ones(3,3);
    A(K,K) = A(K,K) + Ae;
end
