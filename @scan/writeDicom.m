function aScan = writeDicom(aScan, outputFolder, img)
% aScan.writeDicom(outputFolder, img) writes the image data img to DICOM format and saves the images in directory outputFolder.  The matrix img is assumed to have resolution 1x1x1, and is resampled to the resolution specified by aScan.originalElementSpacing before saving to DICOM.  If no img matrix is provided, aScan.img is written.  The headers from aScan.dicoms are used and modified appropriately.

if nargin < 3
% Get original resolution image
img = aScan.getOriginalImage;
else
img = aScan.getOriginalImage(img);

end

% Create output folder
chkmkdir(outputFolder);

% Dicom UID values for this scan and its associated study
if isempty(aScan.studyUID)
    studyUID = dicomuid;
else
    studyUID = aScan.studyUID;
end

if isempty(aScan.seriesUID)
    seriesUID = dicomuid;
else
    seriesUID = aScan.seriesUID;
end

if isempty(aScan.studyDescription)
    aScan.studyDescription = sprintf('5D Clinical Protocol');
    warning('No StudyDescription tag for this scan.  Writing default.')

end

if isempty(aScan.seriesDescription)
    aScan.seriesDescription = '5D Scan';
    warning('No SeriesDescription tag for this scan.  Writing default.')
end

% Flip image if necessary
if aScan.direction == 0
    img = flipdim(img,3);
    zPositions = flipdim(aScan.zPositions,1);
else
    zPositions = aScan.zPositions;
end

%


sliceHeader = dicominfo(aScan.dicoms{1});

% Undo scaling and intercept adjustment made when reading in dicoms 
img = (img  - sliceHeader.RescaleIntercept) ./ sliceHeader.RescaleSlope;

saveBar = waitbar(0,'Saving image...');
% Write slices to dicom


for j = 1:size(img,3)


    sliceHeader = dicominfo(aScan.dicoms{j});

    % Modify tags	
	sliceName = sprintf('slice%03d.dcm',j);
    sliceHeader.PatientPosition = 'HFS';
    sliceHeader.SOPInstanceUID = dicomuid;
	sliceHeader.SeriesInstanceUID = seriesUID;
    sliceHeader.StudyInstanceUID = studyUID;
	sliceHeader.StudyDescription = sprintf(aScan.studyDescription);
    sliceHeader.SeriesDescription = sprintf(aScan.seriesDescription);


	% Save
	slice = squeeze(img(:,:,j));
	dicomwrite(int16(slice),fullfile(outputFolder, sliceName),sliceHeader);
		
        try
		waitbar(j/size(img,3),saveBar);
        end
    end


try
	close(saveBar);
end

if nargin < 3
aScan.save;
end
