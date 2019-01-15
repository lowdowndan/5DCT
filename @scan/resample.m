function aScan = resample(aScan, newElementSpacing)

% Resample to 1x1x1 if no argument is given
if (~exist('newElementSpacing','var'))
	newElementSpacing = [1 1 1];
end


% Get element spacing and dimension.
elementSpacing = aScan.elementSpacing;
dim = aScan.dim;

% Is this an original scan?
if(aScan.original)

    % If not resampling to 1x1x1, change the filename
    if(~isequal(newElementSpacing(:),ones(3,1)))
    aScan.filename = fullfile(fileparts(aScan.filename),sprintf('%02d_resampled_%dx%dx%0.1f.mat',aScan.number,newElementSpacing(1),newElementSpacing(2),newElementSpacing(3)));
    end
    
    if(isempty(aScan.img))
    img = aScan.get_image;
    else
    img = aScan.img;
    end
    
else
    img = aScan.img;
end

%% Resample image

% Verify that DICOM RCS X coordinate increases with image rows, and Y with
% columns.  Vomit and quit if this is not the case.
if(~isempty(aScan.dicoms))
header = dicominfo(aScan.dicoms{1});
assert(isequal(header.ImageOrientationPatient(1),1) & isequal(header.ImageOrientationPatient(5),1), 'ImageOrientationPatient DICOM tag does not match expected value.')
end

% Original grid
xx = aScan.imagePositionPatient(1) : elementSpacing(1) : aScan.imagePositionPatient(1) + ((dim(1) - 1) * elementSpacing(1));
yy = aScan.imagePositionPatient(2) : elementSpacing(2) : aScan.imagePositionPatient(2) + ((dim(2) - 1) * elementSpacing(2));
zz = aScan.zPositions;

[XX,YY,ZZ] = meshgrid(xx,yy,zz);

% Extent of grid in mm
imgExtent= zeros(3,1);
imgExtent(1) = dim(1) * elementSpacing(1); 
imgExtent(2) = dim(2) * elementSpacing(2); 
imgExtent(3) = dim(3) * elementSpacing(3); 

% ImagePositionPatient gives the coordinates of the CENTER of the first voxel.
% Because we're changing the pixel spacing, we're also changing the position of
% the first voxel's center.  In order to prevent inducing a shift, we have to
% change ImagePositionPatient as follows: from ImagePositionPatient, step (0.5
% * element spacing) back to reach the edge of the image, then step (0.5 * new
% element spacing) forward to get new voxel center position.  This ensures that
% the image coverage is the exact same, and there is no shift induced.


newImagePositionPatient = nan(3,1);
newImagePositionPatient(1) = aScan.imagePositionPatient(1) - (aScan.elementSpacing(1) / 2) + newElementSpacing(1)/2;
newImagePositionPatient(2) = aScan.imagePositionPatient(2) - (aScan.elementSpacing(2) / 2) + newElementSpacing(2)/2;

xi = newImagePositionPatient(1) : newElementSpacing(1) : newImagePositionPatient(1) + ((imgExtent(1) - newElementSpacing(1)) * newElementSpacing(1));
yi = newImagePositionPatient(2) : newElementSpacing(2) : newImagePositionPatient(2) + ((imgExtent(2) - newElementSpacing(2)) * newElementSpacing(2));


% Do the same for the z coordinate
oldIncrement = sign(aScan.zPositions(2) - aScan.zPositions(1)) * elementSpacing(3);
newIncrement = sign(aScan.zPositions(2) - aScan.zPositions(1)) * newElementSpacing(3);

zStart = aScan.zPositions(1) - (oldIncrement / 2) + (newIncrement / 2);
%zEnd = aScan.zPositions(end) - (aScan.elementSpacing(3) / 2) + (newElementSpacing(3) / 2); 
zEnd = aScan.zPositions(end) + (oldIncrement / 2) - (newIncrement / 2); 
%zEnd = aScan.zPositions(end); 

zi = zStart: newIncrement : zEnd;
%keyboard
[XI,YI,ZI] = meshgrid(xi,yi,zi);
img = interp3(XX,YY,ZZ,img,XI,YI,ZI,'linear',-1024);

%% Resample surrogates if this is a free-breathing scan
if(numel(aScan.v) > 1)
v = interp1(zz,aScan.v,zi,'spline','extrap');
f = interp1(zz,aScan.f,zi,'spline','extrap');
ekg = interp1(zz,aScan.ekg,zi,'spline','extrap');


	% Resample timepoints if this is an original scan (otherwise t will be empty)
	if(numel(aScan. t) > 1)
	t = interp1(zz,aScan.t, zi,'linear','extrap');
	aScan.t = t(:);
	end

aScan.v = v(:);
aScan.f = f(:);
aScan.ekg = ekg(:);
end


%% Update scan object
aScan.originalDim = dim;
aScan.originalElementSpacing = elementSpacing;
aScan.originalZPositions = aScan.zPositions;
aScan.originalImagePositionPatient = aScan.imagePositionPatient;

aScan.img = img;
aScan.dim = size(aScan.img);
aScan.elementSpacing = newElementSpacing;
aScan.zPositions = zi;
aScan.imagePositionPatient = newImagePositionPatient;

end

