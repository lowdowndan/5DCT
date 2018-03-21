
function img = resampleImage(img, dX, dY, dZ, direction)

dim = size(img);
img = single(img);

[xi,yi,zi] = meshgrid((dX/2):dX:dim(1) - (dX/2), (dY/2):dY:dim(2) - (dY/2), (dZ/2):dZ:dim(3) - (dZ/2));
%[xi,yi,zi] = meshgrid(1:dY:dim(2), 1:dX:dim(1), 1:dZ:dim(3));

xi = single(xi);
yi = single(yi);
zi = single(zi);

% Check direction if povided
if (nargin > 4)
    if direction == 0
    zi = flipdim(zi,3);
    end
end
%[X,Y,Z] = meshgrid(1:size(img,1),1:size(img,2),1:size(img,3));
img = trilinterp(img, xi, yi, zi);
%img = interp3(X,Y,Z,img,xi,yi,zi);
