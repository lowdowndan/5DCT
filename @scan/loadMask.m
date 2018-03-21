function mask = loadMask(aScan,maskFolder)
% mask = aScan.loadMask(maskFolder) reads in a segmentation saved as a collection of DICOMS in maskFolder, assuming the same geometry as the scan object, and resampled the segmentation to 1x1x1.


[img,headers] = dicomfolder(maskFolder);
% %%TODO: fix sorting slices
% files = dir(maskFolder);
% files = setdiff({files.name},{'.','..'});
% 
% img = dicomread(fullfile(maskFolder,files{1}));
% header = dicominfo(fullfile(maskFolder,files{1}));
% img = zeros([size(img) length(files)], 'single');


if iscell(headers)
header = headers{1};
elseif istruct(headers)
header = headers(1);
else
error('Invalid DICOM headers');
end

elementSpacing = zeros(1,3);
elementSpacing(1:2) = header.PixelSpacing;
elementSpacing(3) = header.SliceThickness;

% for iSlice = 1 :length(files);
%     
%     if iSlice == 1
%     	header = dicominfo(fullfile(maskFolder, files{iSlice}));
%         slope = header.RescaleSlope;
%         intercept = header.RescaleIntercept;
%     end
% 
%      img(:,:,iSlice) = dicomread(fullfile(maskFolder, files{iSlice}));
%     
% end

mask = resampleImage(img, 1 / elementSpacing(1), 1 / elementSpacing(2), 1 / elementSpacing(3), aScan.direction);

%mask = (mask * slope) + intercept;

mask(mask > 0.5) = 1;
mask(mask < 1) = 0;

if ~aScan.direction
    mask = flipdim(mask,3);
end
