function aScan = write_dicom(aScan, outputFolder)

%% Create output folder
chkmkdir(outputFolder);

%% Dicom UID values for this scan and its associated study
if isempty(aScan.studyUID)
    aScan.studyUID = dicomuid;
    studyUID = aScan.studyUID;
    aScan.save;
else
    studyUID = aScan.studyUID;
end

%% Study and series descriptions

if isempty(aScan.studyDescription)
    aScan.studyDescription = sprintf('5D Clinical Protocol');
    warning('No StudyDescription tag for this scan.  Writing default.')
end

if isempty(aScan.seriesDescription)
    aScan.seriesDescription = '5D Scan';
    warning('No SeriesDescription tag for this scan.  Writing default.')
end

%% Load image

if(isempty(aScan.img) && aScan.original)
    
img = aScan.get_image;
else

assert(~isempty(aScan.img), 'Missing image, cannot write to DICOM.');    
img = aScan.img;
end

%saveBar = waitbar(0,'Saving image...');
% Write slices to dicom

% Template for DICOM header information
sliceHeader = aScan.acquisitionInfo;
sliceHeader.SeriesInstanceUID = aScan.seriesUID;
sliceHeader.StudyInstanceUID = aScan.studyUID;
sliceHeader.StudyDescription = char(aScan.studyDescription);
sliceHeader.SeriesDescription = char(aScan.seriesDescription);
sliceHeader.PixelSpacing = [1; 1];
sliceHeader.Rows = size(img,1);
sliceHeader.Columns = size(img,2);
sliceHeader.ConversionType = 'WSD';
sliceHeader.SliceThickness = 1;
sliceHeader.ImagePositionPatient = aScan.imagePositionPatient;


% Modify manufacturer tag if scan is dervived

if(~aScan.original)
    %sliceHeader.Manufacturer = '5DCT';
end
    
% Undo scaling and intercept adjustment made when reading in dicoms 
img = (img  - sliceHeader.RescaleIntercept) ./ sliceHeader.RescaleSlope;


for j = 1:size(img,3)

    % Modify tags	
    sliceHeader.SOPInstanceUID = dicomuid;
    sliceHeader.SliceLocation = aScan.zPositions(j);
    sliceHeader.ImagePositionPatient(3) = aScan.zPositions(j);

	% Save
	slice = squeeze(img(:,:,j));
	dicomwrite(int16(slice),fullfile(outputFolder, [sliceHeader.SOPInstanceUID '.dcm']),sliceHeader,'CreateMode','create');
		
        try
		%waitbar(j/size(img,3),saveBar);
        end
    end


try
	%close(saveBar);
end

