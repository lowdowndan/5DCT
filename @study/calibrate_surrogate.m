function aStudy = calibrate_surrogate(aStudy, dicomTable, refScan)
   

load('fivedcolor');
scanIDs = aStudy.scanIDs;
sliceCounts = aStudy.sliceCounts;

surrogateFolder = fullfile(aStudy.folder,'documents','surrogateCalibration');
mkdir(surrogateFolder);

%% Get headers for reference image headers
seriesInds = strcmp(scanIDs(refScan),dicomTable(:,2));
firstSlice = find(seriesInds,1,'first');
pixelSpacing = dicomTable{firstSlice,6};
dim = dicomTable{firstSlice,7};
rescale = dicomTable{firstSlice,8};
sliceFilenames = dicomTable(seriesInds,1);
zPositions = cell2mat(dicomTable(seriesInds,4));
zPositions = zPositions(3:3:end);

% Take only slices within shared scan range
commonRefZpositions = (zPositions > aStudy.scanRange(1) & zPositions < aStudy.scanRange(2));

% Remove slices outside shared range
sliceFilenames = sliceFilenames(commonRefZpositions);
zPositions = zPositions(commonRefZpositions);

%% Load reference image (only shared region)
scanImage = zeros(dim(1),dim(2),length(zPositions),'single');
for jFile = 1:length(zPositions)
scanImage(:,:,jFile) = dicomread(sliceFilenames{jFile});
end

%% Locate profile
[profileRows, profileCol, profileZ] = aStudy.locate_profile(scanImage, zPositions, surrogateFolder);

%% Do scans need to be interpolated?
interpNeeded = false(aStudy.nScans,1);

%% Get abdomen heights
abdomenHeights = zeros(aStudy.nScans,1);

% Suppress warnings
warning('off','stats:nlinfit:ModelConstantWRTParam');
warning('off','MATLAB:rankDeficientMatrix');
warning('off','MATLAB:nearlySingularMatrix');
warning('off','stats:nlinfit:IllConditionedJacobian');


heightBar = waitbar(0,'Finding abdominal heights...');
profileFig = figure;
tight_subplot(1,3,.01);

% Iterate over all scans
for iScan = 1:aStudy.nScans
    
% Get headers 
seriesInds = strcmp(scanIDs(iScan),dicomTable(:,2));
scanTable = dicomTable(seriesInds,:);
zPositions = cell2mat(scanTable(:,4));
zPositions = zPositions(3:3:end);

% Use only slices within range
commonRefZpositions = (zPositions > aStudy.scanRange(1) & zPositions < aStudy.scanRange(2));
scanTable = scanTable(commonRefZpositions, :);
zPositions = zPositions(commonRefZpositions);

[~, profileInd] = min(abs(zPositions - profileZ));
scanProfileZ = zPositions(profileInd);

% Verify that correct slice location was found
%assert(abs(profileZ - scanProfileZ) <= 1.0, sprintf('Cannot locate selected couch location to within 1.0 mm in scan %02d', iScan))

% Interpolation required?
if(abs(profileZ - scanProfileZ) >= .01)
 
interpNeeded(iScan) = true;

assert(~isequal(profileInd,1) && ~isequal(profileInd, length(zPositions)), sprintf('Cannot interpolate slice position for scan %02d, -- missing data on one side.', iScan))

img1 = dicomread(scanTable{profileInd - 1, 1});
header1 = dicominfo(scanTable{profileInd - 1, 1});
img1pos = header1.ImagePositionPatient;

nRows = header1.Rows;
nCols = header1.Columns;
nRows = double(nRows);
nCols = double(nCols);

pixelSpacing = header1.PixelSpacing;

img2 = dicomread(scanTable{profileInd, 1});
header2 = dicominfo(scanTable{profileInd, 1});
img2pos = header2.ImagePositionPatient;

img3 = dicomread(scanTable{profileInd + 1, 1});
header3 = dicominfo(scanTable{profileInd + 1, 1});
img3pos = header3.ImagePositionPatient;

% Rescale
img1 = img1 * rescale(1);
img1 = single(img1);
img1 = img1 + rescale(2);

img2 = img2 * rescale(1);
img2 = single(img2);
img2 = img2 + rescale(2);

img3 = img3 * rescale(1);
img3 = single(img3);
img3 = img3 + rescale(2);

% Original grid
xx = img1pos(1) : pixelSpacing(1) : img1pos(1) + ((nRows - 1) * pixelSpacing(1));
yy = img1pos(2) : pixelSpacing(2) : img1pos(2) + ((nCols - 1) * pixelSpacing(2));
zz = [img1pos(3) img2pos(3) img3pos(3)];

[X,Y,Z] = meshgrid(xx,yy,zz);

% Closest 3 slices to query slice location
img = zeros([size(img1) 3], 'single');
img(:,:,1) = img1;
img(:,:,2) = img2;
img(:,:,3) = img3;

% Interpolation grid
Xi = X(:,:,2);
Yi = Y(:,:,2);
Zi = ones(size(Xi)) * profileZ;

% Get interpolated image at query slice location
imgAx = squeeze(interp3(X,Y,Z, img, Xi, Yi, Zi));

else

imgAx = dicomread(scanTable{profileInd,1});
imgAx = single(imgAx);
imgAx = imgAx * rescale(1);
imgAx = imgAx + rescale(2);
end

imgProfile = imgAx(profileRows(1):profileRows(2), profileCol);

% Show HU profile
subplot(1,3,1);
plot([profileRows(1):profileRows(2)], imgProfile, '--.', 'color', fivedcolor.blue);
set(gca,'fontsize',14);
ylabel('HU')
xlabel('Row');
ylim([-1024 400]);

% Show image location
subplot(1,3,2);
imagesc(imgAx);
set(gca,'fontsize',14);
set(gca,'xticklabel',[],'yticklabel',[]);
axis image;
colormap gray;
title('');
hold on
plot([profileCol profileCol], [profileRows(1), profileRows(2)],'color',fivedcolor.red,'linewidth',1.5);
hold off

drawnow;
pause(.01);
	
set(profileFig, 'units','normalized');
set(profileFig, 'position', [0.0323    0.1783    0.8849    0.5108]);
set(profileFig, 'outerposition', [0.0323    0.1783    0.8849    0.5108]);
set(profileFig,'color',[1 1 1]);

% Find the jump in HU values corresponding to border of body/air
if(iScan == 1)
subplot(1,3,3);
param = sigm_fit([profileRows(1) : profileRows(2)], imgProfile,[],[],1);
initialParam = param;
else
subplot(1,3,3);
param = sigm_fit([profileRows(1) : profileRows(2)], imgProfile,[],initialParam,1);
set(gca,'fontsize',14);
ylabel('HU')
xlabel('Row');
ylim([-1024 400]);
end

abdomenHeights(iScan) = (dim(1) - param(3)) * pixelSpacing(1);

subplot(1,3,3)
hold on
plot([param(3) param(3)], get(gca,'ylim'), '--','color','k','linewidth',1.5);
hold off

f = getframe(gcf);
imwrite(f.cdata,fullfile(surrogateFolder,sprintf('%02d.png',iScan)),'png');

subplot(1,3,2)
set(gca,'fontsize',24)
title(sprintf('Scan %02d', iScan));

try
waitbar(iScan/aStudy.nScans, heightBar);
end

end

% Restore warnings
warning('on','stats:nlinfit:ModelConstantWRTParam');
warning('on','MATLAB:rankDeficientMatrix');
warning('on','MATLAB:nearlySingularMatrix');
warning('on','stats:nlinfit:IllConditionedJacobian');


try
	close(heightBar);
	close(profileFig);
end

% Save abdomen height results
aStudy.abdomenHeights = abdomenHeights;

%% Get voltage values and corresponding times

[scanBellowsVoltage, scanBellowsFlow, scanBellowsTime, scanEkg] = getDataSegments(aStudy);


calibrationVoltages = zeros(size(abdomenHeights),'single');
calibrationTimes = zeros(size(abdomenHeights),'single');

for iScan = 1:aStudy.nScans
   
% Get headers for this scan 
seriesInds = strcmp(scanIDs(iScan),dicomTable(:,2));
acquisitionTimes = cell2mat(dicomTable(seriesInds,3));
scanDuration = acquisitionTimes(end) - acquisitionTimes(1);
zPositions = cell2mat(dicomTable(seriesInds,4));
zPositions = zPositions(3:3:end);

% Account for x-ray warm up.
xrayWarmupDelay = (((aStudy.stopScan(iScan) - aStudy.startScan(iScan))) * (aStudy.sampleRate)) - abs(scanDuration);
xrayWarmup = ceil((xrayWarmupDelay * (1/aStudy.sampleRate)) / 2);

% Symmetric warmup
bellowsVoltageXrayOn = scanBellowsVoltage(xrayWarmup  : end - xrayWarmup, iScan);
bellowsTimeXrayOn = scanBellowsTime(xrayWarmup : end - xrayWarmup, iScan);

% Discard NaN
nanMask = isnan(bellowsVoltageXrayOn) + isnan(bellowsTimeXrayOn);
nanMask = logical(nanMask);

bellowsVoltageXrayOn(nanMask) = [];
bellowsTimeXrayOn(nanMask) = [];

% Normalize bellows and acquisition times
[~, profileInd] = min(abs(zPositions - profileZ));

if(interpNeeded(iScan))
    
    sliceTime = interp1([zPositions(profileInd - 1) zPositions(profileInd) zPositions(profileInd + 1)], [acquisitionTimes(profileInd - 1) acquisitionTimes(profileInd) acquisitionTimes(profileInd + 1)], profileZ);
else
    
   % Get index of the axial slice corresponding to profile location
    sliceTime = acquisitionTimes(profileInd);
end

acquisitionTimeNorm = sliceTime - acquisitionTimes(1);
bellowsTimeXrayOnNorm = bellowsTimeXrayOn - bellowsTimeXrayOn(1);

% Interpolate to find bellows time corresponding to slice acquisition time
calibrationVoltages(iScan) = interp1(bellowsTimeXrayOnNorm,bellowsVoltageXrayOn,acquisitionTimeNorm,'spline');
calibrationTimes(iScan) = interp1(bellowsTimeXrayOnNorm,bellowsTimeXrayOn,acquisitionTimeNorm,'linear');
end

% Save calibration results
aStudy.calibrationVoltages = calibrationVoltages;
aStudy.calibrationTimes = calibrationTimes;
aStudy.initialCorrelation = corr(calibrationVoltages,abdomenHeights);

end

