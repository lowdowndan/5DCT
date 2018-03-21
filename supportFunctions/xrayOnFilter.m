function xrayOnFiltered = xrayOnFilter(xrayOn,sz,sigma)

%sigma = 83;
%sz = 350;
x = linspace(-sz/2,sz/2,sz);
dGauss = (-x ./ (sqrt(2 .* pi .* sigma^3))) .* exp( -x.^2 ./ (2 .* sigma ^2));

xrayOnFiltered = conv(xrayOn,dGauss,'same');
%xrayOnFiltered(1:round(sz/2)) = 0;
%xrayOnFiltered(end-round(sz/2):end) = 0;
xrayOnFiltered = mat2gray(xrayOnFiltered);
