function aScan = resample2(aScan)
% aScan.resample resamples the image stored in aScan.img to 1x1x1.

% Resample to 1x1x1

% Check if image is already correct resolution
if isequal(aScan.elementSpacing(:),ones(3,1))
	return;
else


% Get element spacing and dimension.
elementSpacing = aScan.elementSpacing;
dim = aScan.dim;
img = aScan.img;

%% Resample image

% Verify that DICOM RCS X coordinate increases with image rows, and Y with
% columns.  Vomit and quit if this is not the case.
header = dicominfo(aScan.dicoms{1});
assert(isequal(header.ImageOrientationPatient(1),1) & isequal(header.ImageOrientationPatient(5),1), 'ImageOrientationPatient DICOM tag does not match expected value.')


% Original grid
xx = aScan.imagePositionPatient(1) : elementSpacing(1) : aScan.imagePositionPatient(1) + ((dim(1) - 1) * elementSpacing(1));
yy = aScan.imagePositionPatient(2) : elementSpacing(2) : aScan.imagePositionPatient(2) + ((dim(2) - 1) * elementSpacing(2));
zz = aScan.zPositions;

[XX,YY,ZZ] = meshgrid(xx,yy,zz);

% New grid
newDim = zeros(3,1);
newDim(1) = dim(1) * elementSpacing(1); 
newDim(2) = dim(2) * elementSpacing(2); 
newDim(3) = dim(3) * elementSpacing(3); 

% ImagePositionPatient gives the coordinates of the CENTER of the first
% voxel.  Because we're changing the pixel spacing, we're also changing the
% position of the first voxel's center.  In order to prevent inducing a
% shift, we have to change ImagePositionPatient as follows: from
% ImagePositionPatient, step (0.5 * element spacing) back to reach the edge of
% the image, then step 0.5 forward to get new voxel center position.  This
% ensures that the image coverage is the exact same, and there is no shift
% induced.

newImagePositionPatient = nan(3,1);
newImagePositionPatient(1) = aScan.imagePositionPatient(1) - (aScan.elementSpacing(1) / 2) + 0.5;
newImagePositionPatient(2) = aScan.imagePositionPatient(2) - (aScan.elementSpacing(2) / 2) + 0.5;

xi = newImagePositionPatient(1) : 1 : newImagePositionPatient(1) + ((newDim(1) - 1) * 1);
yi = newImagePositionPatient(2) : 1 : newImagePositionPatient(2) + ((newDim(2) - 1) * 1);

increment = sign(aScan.zPositions(2) - aScan.zPositions(1));

% Do the same for the z coordinate
zStart = aScan.zPositions(1) - (aScan.elementSpacing(3) / 2) + 0.5;
zEnd = aScan.zPositions(end) - (aScan.elementSpacing(3) / 2) + 0.5; 

zi = zStart: increment : zEnd;

[XI,YI,ZI] = meshgrid(xi,yi,zi);

img = interp3(XX,YY,ZZ,img,XI,YI,ZI,'linear', -1024);

%% Resample surrogates
v = interp1(zz,aScan.v,zi,'spline','extrap');
f = interp1(zz,aScan.f,zi,'spline','extrap');
ekg = interp1(zz,aScan.ekg,zi,'spline','extrap');
t = interp1(zz,aScan.t, zi,'linear','extrap');

aScan.v = v(:);
aScan.f = f(:);
aScan.ekg = ekg(:);
aScan.t = t(:);

%% Update scan object
aScan.originalDim = dim;
aScan.originalElementSpacing = elementSpacing;
aScan.originalZPositions = aScan.zPositions;
aScan.originalImagePositionPatient = aScan.imagePositionPatient;

aScan.img = img;
aScan.dim = size(aScan.img);
aScan.elementSpacing = [1 1 1];
aScan.zPositions = zi;
aScan.imagePositionPatient = newImagePositionPatient;

end
end

