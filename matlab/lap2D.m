function A = lap2D(nx,ny)
%% Generates a 2-D Laplacian

tx = sptridiag(-1, 2, -1, nx);
ty = sptridiag(-1, 2, -1, ny);
A = kron(speye(ny,ny),tx) + kron(ty,speye(nx,nx));

end
