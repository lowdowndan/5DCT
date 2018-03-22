%% Generate lung segmentation using pulmonary toolkit

function mask = ptk_mask(aStudy, scanNo)

if(~exist('scanNo','var'))
    scanNo = 1;
    warning('No scan number specified.  Returning mask for scan 01.');
end

%% Load scan

aScan = aStudy.getScan(scanNo);

%% Write scan to dicom temporarily

dcmDir = fullfile(aStudy.folder,'tmpdcm');
mkdir(dcmDir);
aScan.write_dicom(dcmDir);

%% Import dicom files into PTK (clunky; TODO: modify PTK import to take raw images)
PTKAddPaths;
ptk_main = PTKMain;

fileInfos = PTKDicomUtilities.GetListOfDicomFiles(dcmDir);
dataset = ptk_main.CreateDatasetFromInfo(fileInfos);
%lobes = dataset.GetResult('PTKLobes');
lobes = dataset.GetResult('PTKLeftAndRightLungs');

%% Re-grid mask so that it is in original geometry (PTK crops and rescales, not sure how to disable)

mask = lobes.RawImage;
[xm,ym,zm] = lobes.GetDicomCoordinates;
[XM, YM, ZM] = meshgrid(xm,ym,zm);

%% New grid
xi = aStudy.imagePositionPatient(1) : 1 : aStudy.imagePositionPatient(1) + ((aStudy.dim(1) - 1) * 1);
yi = aStudy.imagePositionPatient(2) : 1 : aStudy.imagePositionPatient(2) + ((aStudy.dim(2) - 1) * 1);
zi = aStudy.zPositions;

[XI,YI,ZI] = meshgrid(xi,yi,zi);


%% Interpolate image
mask = interp3(XM,YM,ZM,mask,XI,YI,ZI,'nearest',0);
mask = logical(mask);

%% Save
save(fullfile(aStudy.folder,sprintf('mask_%02d.mat',scanNo)),'mask');

%% Remove directory
rmCmd = ['rm -r "' dcmDir '"'];
system(rmCmd);

%% Remove dataset from PTK
ptk_main.DeleteCacheForAllDatasets;

