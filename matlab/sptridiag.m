function T = sptridiag(a, b, c, n)
% creates tridiagonal matrix with constants a, b, c respectively on 3 diagonals

e = ones(n,1);
T = spdiags([a*e, b*e, c*e], -1:1, n, n);

end
