nx = 20;
nr = 4;

theta = [0:1/nx:1] * pi;
r = [0.4:0.8/nr:1];

% Construct grid of points
X = kron(r,cos(theta));
Y = kron(r, sin(theta));
XY = [X',Y'];

% Counter for number of triangles
t = 0;

for i = 1:nx
    for j = 0:nr-2
        t = t+1;
        tri(t,:) = [i+j*(nx+1), i+j*(nx+1)+1, i+(j+1)*(nx+1)+1];

        t = t+1;
        tri(t,:) = [i+j*(nx+1), i+(j+1)*(nx+1)+1, i+(j+1)*(nx+1)];
    end
end

A = assembl(tri,XY);

gplot(A,XY);
