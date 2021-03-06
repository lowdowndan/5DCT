function shift = align(aScan, refZpositions, refImagePositionPatient, refElementSpacing)
% aScan.align(refZpositions) interpolates the image data at Z locations refZpositions.  This method is called when the couch positions of a scan differ from those of the reference image.  It is necessary to avoid biasing the model parameters with an artificial shift.

shift = zeros(1,3);
zPositions = aScan.zPositions;
v = aScan.v;
f = aScan.f;
ekg = aScan.ekg;
t = aScan.t;
dicoms = aScan.dicoms;

img = aScan.img;

%% Check spacing
if(~isequal(refElementSpacing,aScan.elementSpacing))
    warning('Element spacing of referensce scan is %0.4f x %0.4f x %0.4f, but scan %02d has an element spacing of %0.4f x %0.4f x %0.4f.  Interpolating to correct.', ...
        refElementSpacing(1), refElementSpacing(2), refElementSpacing(3), aScan.number, aScan.elementSpacing(1), aScan.elementSpacing(2), ...
        aScan.elementSpacing(3));
end

%% Original grid
xx = aScan.imagePositionPatient(1) : aScan.elementSpacing(1) : aScan.imagePositionPatient(1) + ((aScan.dim(1) - 1) * aScan.elementSpacing(1));
yy = aScan.imagePositionPatient(2) : aScan.elementSpacing(2) : aScan.imagePositionPatient(2) + ((aScan.dim(2) - 1) * aScan.elementSpacing(2));
zz = aScan.zPositions;

[XX,YY,ZZ] = meshgrid(xx,yy,zz);

%% New grid
xi = refImagePositionPatient(1) : refElementSpacing(1) : refImagePositionPatient(1) + ((aScan.dim(1) - 1) * refElementSpacing(1));
yi = refImagePositionPatient(2) : refElementSpacing(2) : refImagePositionPatient(2) + ((aScan.dim(2) - 1) * refElementSpacing(2));
zi = refZpositions;

[XI,YI,ZI] = meshgrid(xi,yi,zi);

%% Warn if there is a shift in x, y or z.

% X
if (~isequal(aScan.imagePositionPatient(1), refImagePositionPatient(1)))
    shift(1) = refImagePositionPatient(1) - aScan.imagePositionPatient(1);
    warning('Scan %02d is shifted from the reference scan by %0.4f mm in the X direction.  Interpolating to correct.', aScan.number, shift(1));
end

% Y
if (~isequal(aScan.imagePositionPatient(2), refImagePositionPatient(2)))
    shift(2) = refImagePositionPatient(2) - aScan.imagePositionPatient(2);
    warning('Scan %02d is shifted from the reference scan by %0.4f mm in the Y direction.  Interpolating to correct.', aScan.number, shift(2));
end

% Z
if (~isequal(aScan.zPositions(1), refZpositions(1)))
    shift(3) = refZpositions(1) - aScan.zPositions(1);
    warning('Scan %02d is shifted from the reference scan by %0.4f mm in the Z direction.  Interpolating to correct.', aScan.number, shift(3));
end

%% Interpolate image

img = interp3(XX,YY,ZZ,img,XI,YI,ZI,'linear',-1024);


%% Interpolate bellows (Z only)
v = interp1(zPositions,v,refZpositions,'spline','extrap');
f = interp1(zPositions,f,refZpositions,'spline','extrap');
ekg = interp1(zPositions,ekg,refZpositions,'spline','extrap');
t = interp1(zPositions,t, refZpositions,'linear','extrap');

%% Write results
aScan.img = img;
aScan.dim = size(img);
aScan.v = v;
aScan.f = f;
aScan.ekg = ekg;
aScan.t = t;
aScan.zPositions = refZpositions;
aScan.imagePositionPatient = refImagePositionPatient;
aScan.elementSpacing = refElementSpacing;
aScan.dicoms = dicoms;

end
