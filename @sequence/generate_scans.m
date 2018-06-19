%% Generate a sequence of breathing gated images

function generate_scans(aSequence)

% Verify that the reconstruction points have been set
assert(isequal(numel(aSequence.reconstructionPoints), numel(aSequence.inhaleAmplitudes) + numel(aSequence.exhaleAmplitudes)),'Reconstruction points have not been set.  Run set_reconstruction_points method.');


% Load parameters
[aX, aY, aZ] = aSequence.model.get_alpha;
[bX, bY, bZ] = aSequence.model.get_beta;
[cX, cY, cZ] = aSequence.model.get_constant;

% Pre-allocate (1 for each recon point, then error and reference)
% Order: 1 to N: 4DCT Phases; N + 1: MIP; N + 2 Error; N + 3 Reference

aSequence.scans = cell(numel(aSequence.reconstructionPoints) + 3, 1);
aSequence.dicomFolders = cell(numel(aSequence.reconstructionPoints) + 3, 1);


%% Generate images and make scan objects
for iScan = 1: numel(aSequence.reconstructionPoints)
    
    % Image
    v = aSequence.reconstructionPoints(iScan).v;
    f = aSequence.reconstructionPoints(iScan).f;

    img = aSequence.model.generate_image(v,f,aX,aY,aZ,bX,bY,bZ,cX,cY,cZ);
    
    if(iScan == 1)
    mip = zeros(size(img),'single');
    mip = mip - 1024;
    end
    
    
   % function aScan = set_derived(aScan, img, v, f, studyUID, seriesDescription)

    % Scan object
    aScan = scan(aSequence, aSequence.model.study.acquisitionInfo, aSequence.model.study.imagePositionPatient, aSequence.model.study.zPositions);
    aScan.set_derived(img, v, f, aSequence.studyUID, aSequence.reconstructionPoints(iScan).description);
    aScan.save;
    
    % Reference
    aSequence.scans{iScan} = aScan.filename;
    
    % Write dicom
    outDir = fullfile(aSequence.folder,aSequence.reconstructionPoints(iScan).description);
    mkdir(outDir);  
    
    aSequence.dicomFolders{iScan} = outDir;
    aScan.write_dicom(outDir);
    
    % Contribute to MIP
    mip = cat(4,mip,img);
    mip = max(mip,[],4);
                     
end


%% Generate MIP
aScan = scan(aSequence, aSequence.model.study.acquisitionInfo, aSequence.model.study.imagePositionPatient, aSequence.model.study.zPositions);
aScan.set_derived(mip,[],[],aSequence.studyUID,'Maximum Intensity Projection (MIP)');
aScan.set('filename',fullfile(aSequence.folder,'mip.mat'));


aScan.save;

outDir = fullfile(aSequence.folder,'mip');
aScan.write_dicom(outDir);

% Append to scan list
aSequence.scans{numel(aSequence.reconstructionPoints) + 1} = aScan.filename;
aSequence.dicomFolders{numel(aSequence.reconstructionPoints) + 1} = outDir;


%% Generate error image
if(exist(fullfile(aSequence.model.folder,'error.mat'),'file'))
    % load error image
else
    warning('Model error not found.  Using mean residual.');
    error = aSequence.model.get_mean_residual;
end


% Deform to 0% (find minimum amplitude)
[~, minPhase] = min([aSequence.reconstructionPoints.amplitude]);
    
v = aSequence.reconstructionPoints(minPhase).v;
f = aSequence.reconstructionPoints(minPhase).f;
error = aSequence.model.deform_image(error,v,f,aX,aY,aZ,bX,bY,bZ,cX,cY,cZ);
    
% Error scan
aScan = scan(aSequence, aSequence.model.study.acquisitionInfo, aSequence.model.study.imagePositionPatient, aSequence.model.study.zPositions);
% errorDesc = aSequence.reconstructionPoints(minPhase).description;
errorDesc = 'Error Image End Exhale';
    
aScan.set_derived(error, v, f, aSequence.studyUID,errorDesc);
aScan.set('filename',fullfile(aSequence.folder,'error.mat'));
aScan.save;
    
outDir = fullfile(aSequence.folder,'error');
mkdir(outDir);  
aScan.write_dicom(outDir);     
    
% Append to scan list
aSequence.scans{numel(aSequence.reconstructionPoints) + 2} = aScan.filename;
aSequence.dicomFolders{numel(aSequence.reconstructionPoints) + 2} = outDir;

%% Write reference image (free-breathing)

aScan = aSequence.model.study.get_scan(aSequence.model.registration.refScan);
outDir = fullfile(aSequence.folder,'ref');
mkdir(outDir); 
aScan.write_dicom(outDir);

% Append to scan list
aSequence.scans{numel(aSequence.reconstructionPoints) + 3} = aScan.filename;
aSequence.dicomFolders{numel(aSequence.reconstructionPoints) + 3} = outDir;
    
end
