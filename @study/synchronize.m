function aStudy = synchronize(aStudy, refScan)

%% Verify that study has not already been synchronized;

if(strcmp(aStudy.status,'synchronized'))
   % error('This study has already been synchronized.  Overwriting synchronized data is not allowed.  Please create a new study.');
end


%% Default reference scan is 1
% TODO:
% Add reference scan selection

if(~exist('refScan','var'))
refScan = 1;
end

%% Select data channels
if(isempty(aStudy.channels))
	aStudy.setChannels;
end

%% Get inhalation direction, flip if necessary

if(isempty(aStudy.bellowsInhaleDirection))
    aStudy.set_bellows_inhale_direction;
end
   

%% Smooth voltage signal
if(isempty(aStudy.bellowsSmoothingWindow))
aStudy.data(:,aStudy.channels.voltage) = study.smooth(aStudy.data(:,aStudy.channels.voltage));
aStudy.bellowsSmoothingWindow = 51;
end

%% Set sample rate
if(isempty(aStudy.sampleRate))
    aStudy.setSampleRate;
end

%% Prompt user for relevant region of breathing trace
if(isempty(aStudy.dataRange))
    aStudy.setDataRange;
    aStudy.patient.save;
end

%% Get scan start/end times
if(isempty(aStudy.startScan) || isempty(aStudy.stopScan))
 
     aStudy.setScanSegments;
     % Plot and record scan segments
     aStudy.plotScanSegments(1);
     aStudy.patient.save;

end

%% Get table of dicom tags:
% Table is structured as follows:
% | Filename | SeriesUID | Acquisition Time | Image Position Patient [X,Y,Z] | Slice Thickness | [X Spacing; Y Spacing] | [Rows; Columns] | ...
% [rescale slope; rescale intercept] |SOP Instance UID

% Save dicom table?
if exist(fullfile(aStudy.folder,'dicomTable.mat'))
load(fullfile(aStudy.folder,'dicomTable.mat'));
else
dicomTable = get_dicomTable(aStudy);
save(fullfile(aStudy.folder,'dicomTable.mat'),'dicomTable');
end

%% Parse dicom table and verify all scans have the same Z positions, pixel spacing, etc 
%[dicomTable, scanIDs, sliceCounts] = aStudy.processDicomTable(dicomTable);

if(isempty(aStudy.scanIDs) || isempty(aStudy.sliceCounts))
dicomTable = aStudy.verify_image_positions(dicomTable);

% Save updated dicom table
save(fullfile(aStudy.folder,'dicomTable.mat'),'dicomTable');

% Save patient
aStudy.patient.save;
end

%% Calibrate surrogate
if(isempty(aStudy.abdomenHeights) || isempty(aStudy.calibrationVoltages) || isempty(aStudy.calibrationTimes) || isempty(aStudy.drift))

% Initial estimate of 0 drift
calibrated = false;

while(calibrated == false)
  
% Calibrate    
%% TODO:
% Add interpolation if slice doesnt exist in all scans
aStudy.calibrate_surrogate(dicomTable, refScan);

% Report initial correlation
linearityFig = figure;
plot(aStudy.abdomenHeights,aStudy.calibrationVoltages,'k+','markersize',8,'linewidth',1.5);
set(linearityFig,'units','normalized');
set(linearityFig,'Position',[0         0    0.99    0.99]);
set(gca,'fontsize',24);
xlabel('Abdominal height (mm)');
ylabel('Bellows (V)');
hold on
surrogateLine = polyfit(aStudy.abdomenHeights,aStudy.calibrationVoltages,1);
hLine = refline(surrogateLine);
hLine.Color = 'r';
r = aStudy.initialCorrelation;
title(sprintf('R = %0.2f',r));
set(linearityFig,'color',[1 1 1]);
% Get confirmation from user
% Dialog box parameters
question = 'Confirm?';
dlgTitle = 'Surrogate calibration';
respNo = 'No';
respYes = 'Yes';

userResponse = questdlg(question,dlgTitle,respYes,respNo,respNo);

if(strcmp(userResponse,respYes))
calibrated = true;
% Drift correct
aStudy.correct_drift;

% Save linearity figure
f = getframe(linearityFig);
imwrite(f.cdata,fullfile(aStudy.folder,'documents','surrogatePlot.png'),'png');


else
close(linearityFig);
end

end
    
end

%% Get data segments that refelct drift correction
[scanBellowsVoltage, scanBellowsFlow, scanBellowsTime, scanEkg] = getDataSegments(aStudy);


%% Generate and store a UID for this study    
aStudy.studyUID = dicomuid;


%% Process scan images

%Initialize waitbar
loadImages = waitbar(0,'Loading, synchronizing, and saving scans...');

% Load images, process reference image first
scanIterate = [refScan, setdiff([1:aStudy.nScans], refScan)];
seriesInds = strcmp(aStudy.scanIDs(1),dicomTable(:,2));
firstSlice = find(seriesInds,1,'first');
lastSlice = find(seriesInds,1,'last');

header = dicominfo(dicomTable{firstSlice,1});
scannerModel = header.ManufacturerModelName;

% Verify scanner model, set correct orientation for coordinate system.
if( strcmp(scannerModel, 'Biograph 64'))

    zOrientation = 1;

elseif (strcmp(scannerModel,'SOMATOM Definition Flash') || strcmp(scannerModel,'SOMATOM Definition AS'))
    
    zOrientation = 1;

else
error('Scanner model not supported.  This toolbox only supports the Siemens Definition Flash, Biograph 64, or Definition AS.');
end


for jScan = 1:aStudy.nScans

iScan = scanIterate(jScan);
% Get metadata for this scan
seriesInds = strcmp(aStudy.scanIDs(iScan),dicomTable(:,2));
firstSlice = find(seriesInds,1,'first');
lastSlice = find(seriesInds,1,'last');

pixelSpacing = dicomTable{firstSlice,6};
sliceThickness = dicomTable{firstSlice,5};
dim = dicomTable{firstSlice,7};
rescale = dicomTable{firstSlice,8};
acquisitionTimes = cell2mat(dicomTable(seriesInds,3));
scanDuration = acquisitionTimes(end) - acquisitionTimes(1);
sliceFilenames = dicomTable(seriesInds,1);

zPositions = cell2mat(dicomTable(seriesInds,4));
% Take only Z coordinate from image position patient vector
zPositions = zPositions(3:3:end);


referenceFrame = dicominfo(dicomTable{firstSlice,1});
referenceFrame = referenceFrame.FrameOfReferenceUID;

% Set acquisition info, if this scan is the reference scan
if(iScan == refScan)

% Set this scans z positions as the reference.  Other scans will be
% interpolated onto these slice locations if necessary.
refZpositions = zPositions;

% Set acquisition date, and info (used for writing to DICOM).
aStudy.set_acquisition_info(dicomTable);

% Set patient name and mrn
%aStudy.setPatientInfo(dicomTable);
else

% Make sure this scan is in the same frame of reference as the reference
% scan.  Otherwise the DICOM coordinates don't match and nothing we are
% about to do makes sense.
assert(strcmp(referenceFrame,aStudy.acquisitionInfo.FrameOfReferenceUID), sprintf('Scan %02d does not have the same frame of reference UID as the reference scan.',iScan));

end

% Load image
scanImage = zeros(dim(1),dim(2),aStudy.sliceCounts(iScan),'single');
for jFile = 1:aStudy.sliceCounts(iScan)
scanImage(:,:,jFile) = dicomread(sliceFilenames{jFile});
end


% Find slices outside of shared range, if any, and flag for removal after
% synchronization
removeMask = zPositions < aStudy.scanRange(1) | zPositions > aStudy.scanRange(2);

% Check scan orientation (head to foot vs foot to head)
if sign((zPositions(end) - zPositions(1))) == zOrientation
	scanDirection = 1;
else
	scanDirection = 0;
end

% Account for x-ray warm up.
xrayWarmupDelay = (((aStudy.stopScan(iScan) - aStudy.startScan(iScan))) * (aStudy.sampleRate)) - abs(scanDuration);
xrayWarmup = ceil((xrayWarmupDelay * (1/aStudy.sampleRate)) / 2);

% Front-loaded warmup
%xrayWarmup = 2 * xrayWarmup;
%bellowsVoltageXrayOn = scanBellowsVoltage(xrayWarmup  : end, iScan);
%bellowsTimeXrayOn = scanBellowsTime(xrayWarmup : end, iScan);
%acqEkg = scanEkg(xrayWarmup: end, iScan);

% Symmetric warmup
bellowsVoltageXrayOn = scanBellowsVoltage(xrayWarmup  : end - xrayWarmup, iScan);
bellowsFlowXrayOn = scanBellowsFlow(xrayWarmup  : end - xrayWarmup, iScan);
bellowsTimeXrayOn = scanBellowsTime(xrayWarmup : end - xrayWarmup, iScan);
acqEkg = scanEkg(xrayWarmup: end - xrayWarmup, iScan);

% Normalize bellows and acquisition times
acquisitionTimesNorm = acquisitionTimes - acquisitionTimes(1);
bellowsTimeXrayOnNorm = bellowsTimeXrayOn - bellowsTimeXrayOn(1);

% Remove nans
nanInds = isnan(bellowsVoltageXrayOn);
bellowsVoltageXrayOn(nanInds) = [];
acqEkg(nanInds) = [];
bellowsTimeXrayOnNorm(nanInds) = [];
bellowsFlowXrayOn(nanInds) = [];

% Interpolate to find bellows time corresponding to slice acquisition time
bellowsVoltageSlices = interp1(bellowsTimeXrayOnNorm,bellowsVoltageXrayOn,acquisitionTimesNorm,'spline');
acqEkgSlices = interp1(bellowsTimeXrayOnNorm,acqEkg,acquisitionTimesNorm,'spline');
flowSlices = interp1(bellowsTimeXrayOnNorm,bellowsFlowXrayOn,acquisitionTimesNorm,'spline');


% Remove slices outside of range, along with surrogate/ekg measurements
if (any(removeMask))

    scanImage(:,:,removeMask) = [];
    bellowsVoltageSlices(removeMask) = [];
    acqEkgSlices(removeMask) = [];
    flowSlices(removeMask) = [];
    acquisitionTimes(removeMask) = [];
    sliceFilenames(removeMask) = [];
    zPositions(removeMask) = [];
    
    % Remove from reference scan
    if (iScan == refScan)
        refZpositions(removeMask) = [];
    end
    
end


% Flip image and voltage if necessary
if scanDirection < 1

imagePositionPatient = dicomTable{firstSlice,4};
bellowsVoltageSlices = flipdim(bellowsVoltageSlices,1);
flowSlices = flipdim(flowSlices,1);
acqEkgSlices = flipdim(acqEkgSlices,1);
acquisitionTimes = flipdim(acquisitionTimes,1);
scanImage = flipdim(scanImage, 3);
zPositions = flipdim(zPositions,1);
sliceFilenames = flipdim(sliceFilenames(:),1);

    % If this is the reference scan, flip the stored z positions in the
    % study object
    if iScan == refScan
        
        % Check if the reference z positions have already been flipped in a
        % previous synchronization attempt.
        if (sign((refZpositions(end) - refZpositions(1))) ~= zOrientation)
                    refZpositions = flipdim(refZpositions,1);
        end
        
    end
    
else
    imagePositionPatient = dicomTable{lastSlice,4};
end


% Ignore the Z value of imagePositionPatient;  It is not necessarily
% accurate as the slice it was taken from may have been removed. The X and Y values
% are all that is used for aligning/interpolation;  Z value is treated
% separately in zPositions/refZpostions variables.
imagePositionPatient(3) = nan;

% Rescale image according to slope and intercept
scanImage = (scanImage .*  rescale(1)) + rescale(2);

% Write scan
aScan = scan(aStudy, aStudy.acquisitionInfo, imagePositionPatient, zPositions);
aScan.setOriginal(scanImage, iScan, [pixelSpacing(:)' sliceThickness], sliceFilenames, ...
    bellowsVoltageSlices, flowSlices, acqEkgSlices, acquisitionTimes, scanDirection, aStudy.studyUID);

% Set the reference image position patient (before resampling to 1x1x1) if this is the
% reference scan
if isequal(iScan, refScan)

   refImagePositionPatient = imagePositionPatient;

else
    
    % If this is not the reference scan, verify that this scan is at the
    % same location as the reference.  Align if necessary.
    if (~isequal(refZpositions,zPositions) || ( ~isequal(refImagePositionPatient(1),imagePositionPatient(1))) || (~isequal(refImagePositionPatient(2),imagePositionPatient(2))))
    aScan = aScan.align2(refZpositions, refImagePositionPatient);
    end
    
end


% Resample to 1x1x1 mm resolution, resample surrogate signals if necessary
% (done if any slice thickness other than 1.0 is used)
aScan.resample2;

% Store data from reference scan in study object

if iScan == refScan

% Pre-allocate for remaining scans
aStudy.scans = cell(aStudy.nScans,1);
aStudy.v = nan(size(aScan.v,1),aStudy.nScans);
aStudy.f = nan(size(aScan.v,1),aStudy.nScans);
aStudy.ekg = nan(size(aScan.v,1),aStudy.nScans);
aStudy.importedScanIDs = cell(aStudy.nScans,1);
% Reference scan info (these are common to all scans after
% verify_image_positions and align methods).  Store imagePositionPatient
% (changed from refImagePositionPatient if resampling was performed).
aStudy.zPositions = aScan.get('zPositions');
aStudy.dim = aScan.dim;
aStudy.imagePositionPatient = aScan.get('imagePositionPatient');

end

% Add description
aScan.seriesDescription = sprintf('%02d of %02d',iScan, aStudy.nScans);
aStudy.importedScanIDs{iScan} = aScan.seriesUID;
% Save scan filename 
aStudy.scans{iScan} = aScan.filename;

% Save synchronized data
aStudy.v(:,iScan) = aScan.v;
aStudy.f(:,iScan) = aScan.f;
aStudy.ekg(:,iScan) = aScan.ekg;
aStudy.direction(iScan) = scanDirection;

% Write image data to .nii for registration
chkmkdir(fullfile(aStudy.folder,'nii'));
aScan.writeNii(fullfile(aStudy.folder,'nii',sprintf('%02d',iScan)));

% Save scan metadata
aScan.img = [];
aScan.save;
delete(aScan);
clear aScan;

try
waitbar((iScan/aStudy.nScans), loadImages);
end
end

try
close(loadImages)
end

% Mark study as synchronized
aStudy.status = 'synchronized';
% Call for save of patient object
aStudy.patient.save;

end


