%% set_representative_slices
%
% Determine the coronal and sagittal slices for displaying images

function set_representative_slices(aRegistration)

%% Load / generate mask for reference scan

mask = aRegistration.study.get_mask(aRegistration.refScan);

%% Find coronal slice

nVoxels = zeros(size(mask,1),1,'single');

for iSlice = 1:size(mask,1)
    
    nVoxels(iSlice) = nnz(squeeze(mask(iSlice,:,:)));
end

[~, corSlice] = max(nVoxels);

%% Find sagittal slice

for iSlice = 1:size(mask,1)
    
    nVoxels(iSlice) = nnz(squeeze(mask(:,iSlice,:)));
end

[~, maxInd] = max(nVoxels);
    
%[mm,~] = peakdet(nVoxels,5000);
sel = (max(nVoxels) - min(nVoxels)) / 4;
mm = peakfinder(nVoxels,sel);

% Too many peaks?
if(size(mm,1) > 2)
mm = peakfinder(nVoxels,(2 * sel));
    
% Not enough?
elseif(size(mm,1) < 2)
mm = peakfinder(nVoxels,(0.01 * sel));
end

% I give up
if(size(mm,1) ~= 2)
    warning('Failed to find representative sagittal slices for left and right lungs.  Using single sagittal slice.');

sagSliceR = maxInd;
sagSliceL = maxInd;

else

% Working as intended    
sagSliceR = mm(1);
sagSliceL = mm(2);
  
end

%% Set slices
aRegistration.corSlice = corSlice;
aRegistration.sagSliceL = sagSliceL;
aRegistration.sagSliceR = sagSliceR;