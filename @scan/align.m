function aScan = align(aScan, refZpositions)
% aScan.align(refZpositions) interpolates the image data at Z locations refZpositions.  This method is called when the couch positions of a scan differ from those of the reference image.  It is necessary to avoid biasing the model parameters with an artificial shift.

zPositions = aScan.zPositions;
v = aScan.v;
f = aScan.f;
ekg = aScan.ekg;
t = aScan.t;
dicoms = aScan.dicoms;

img = aScan.img;


%% Fix unequal number of slices between this scan and the reference scan

% Too few slices?
if numel(zPositions) < numel(refZpositions)
	
% Repeat last value of zPositions.  Interpolation is set
% to clamp on the edge.

% TODO:
% How to handle duplicate dicom headers?

    zPositions(end:length(refZpositions)) = zPositions(end);
    dicoms(end:length(refZpositions)) = dicoms(end);


% Too many slices?
elseif numel(zPositions) > numel(refZpositions)

	nExtraSlices = numel(zPositions) - numel(refZpositions);

	% Which end of the scan extends beyond the reference?
	topOffset = abs(zPositions(1) - refZpositions(1));
	bottomOffset = abs(zPositions(end) - refZpositions(end));

	% Trim at appropriate end
	switch sign(topOffset - bottomOffset)

	case 1
		% Remove slices from top
		zPositions(1:nExtraSlices) = [];
		dicoms(1:nExtraSlices) = [];
	case -1
		% Remove slices from bottom
		zPositions(end - (nExtraSlices - 1): end) = [];
		dicoms(end - (nExtraSlices - 1): end) = [];
	case 0
		% Alternate
		for iSlice = 1: nExtraSlices
			if mod(iSlice,2)
				zPositions(1) = [];
				dicoms(1) = [];
			else
				zPositions(1) = [];
				dicoms(end) = [];
			end
		end
	end
end

% Get shift for each slice
zShift = zPositions - refZpositions;
z = [1:numel(refZpositions)]' + zShift;
% Interpolate image 
[Xi,Yi,Zi] = meshgrid(1:size(img,1),1:size(img,2),z);
Xi = single(Xi);
Yi = single(Yi);
Zi = single(Zi);   
img = trilinterp(img,Xi,Yi,Zi);

% Interpolate bellows
interpZRef = [1:length(v)];
v = interp1(interpZRef,v,z,'linear','extrap');
f = interp1(interpZRef,f,z,'linear','extrap');
ekg = interp1(interpZRef,ekg,z,'linear','extrap');
t = interp1(interpZRef,t, z,'linear','extrap');

% Write results
aScan.img = img;
aScan.v = v;
aScan.f = f;
aScan.ekg = ekg;
aScan.t = t;
aScan.zPositions = zPositions;
aScan.dicoms = dicoms;
end
