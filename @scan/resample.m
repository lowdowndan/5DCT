function aScan = resample(aScan)
% aScan.resample resamples the image stored in aScan.img to 1x1x1.

% Resample to 1x1x1

% Check if image is already correct resolution
if isequal(aScan.elementSpacing(:),ones(3,1));
	return;
else


% Get element spacing and dimension.
elementSpacing = aScan.elementSpacing;
dim = aScan.dim;
img = single(aScan.img);




% Get new spacing
dX = 1/elementSpacing(1);
dY = 1/elementSpacing(2);
dZ = 1/elementSpacing(3);

% Resample
aScan.img = resampleImage(aScan.img,dX,dY,dZ);


% Interpolate bellows if necessary
if aScan.elementSpacing(3) ~= 1

zGrid = sort(aScan.zPositions,1,'ascend');    
zGrid = zGrid - zGrid(1);

zResampled = (0:1:size(aScan.img,3) - 1)';

[~, uniqueInds] = unique(zGrid);
vUnique = aScan.v(uniqueInds);
fUnique = aScan.f(uniqueInds);
tUnique = aScan.t(uniqueInds);
ekgUnique = aScan.ekg(uniqueInds);


%v = interp1(zGrid,aScan.v,zResampled,'spline','extrap');
%f = interp1(zGrid,aScan.f,zResampled,'spline','extrap');
%ekg = interp1(zGrid,aScan.ekg,zResampled,'linear','extrap');
%t = interp1(zGrid,aScan.t, zResampled,'linear','extrap');

zGrid = zGrid(uniqueInds);
v = interp1(zGrid,vUnique,zResampled,'spline',nan);
f = interp1(zGrid,fUnique,zResampled,'spline',nan);
ekg = interp1(zGrid,ekgUnique,zResampled,'linear',nan);
t = interp1(zGrid,tUnique, zResampled,'linear',nan);


v = v(:);
f = f(:);
ekg = ekg(:);
t = t(:);

% Save
aScan.v = v;
aScan.f = f;
aScan.ekg = ekg;
aScan.t = t;
end


aScan.originalElementSpacing = elementSpacing;
aScan.originalDim = dim;
aScan.dim = size(aScan.img);
aScan.elementSpacing = [1 1 1];

end
end

