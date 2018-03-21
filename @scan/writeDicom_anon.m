
function aScan = writeDicom_anon(aScan,img, outputFolder, first, last, id, studyDescription)

if nargin < 6
    studyDescription = '5D Clinical Protocol';
end

if ~exist('img','var')
% Get original resolution image
img = aScan.getOriginalImage;
else
%img = aScan.getOriginalImage(img);
disp('DEBUG: NOT RESAMPLING IMAGE TO ORIGINAL RESOLUTION, BUT WRITING ORIGINAL RESOLUTION IN DICOM HEADER!');

end

% Create output folder
chkmkdir(outputFolder);

% Dicom UID values for this scan and its associated study
if isempty(aScan.studyUID)
    studyUID = dicomuid;
else
    studyUID = aScan.studyUID;
end

%if isempty(aScan.seriesUID)
    seriesUID = dicomuid;
%else
%    seriesUID = aScan.seriesUID;
%end

% Flip image if necessary
if aScan.direction == 0
    img = flipdim(img,3);
    zPositions = flipdim(aScan.zPositions,1);
else
    zPositions = aScan.zPositions;
end

sliceHeader = dicominfo(aScan.dicoms{1});

% Undo scaling and intercept adjustment made when reading in dicoms using scanSync
img = (img  - sliceHeader.RescaleIntercept) ./ sliceHeader.RescaleSlope;

%saveBar = waitbar(0,'Saving image...');
% Write slices to dicom


for j = 1:size(img,3)


    sliceHeader = dicominfo(aScan.dicoms{j});

    % Modify some tags	
	sliceName = sprintf('slice%03d.dcm',j);
    
    sliceHeader.SOPInstanceUID = dicomuid;
	sliceHeader.SeriesInstanceUID = seriesUID;
    sliceHeader.StudyInstanceUID = studyUID;
	sliceHeader.StudyDescription = sprintf(studyDescription);
    sliceHeader.SeriesDescription = aScan.seriesDescription;
    sliceHeader.PatientID = id;
    sliceHeader.PatientName.FamilyName = last;
    sliceHeader.PatientName.GivenName = first;
    sliceHeader.PatientName.MiddleName = '';
    sliceHeader.PatientAddress = '';
    sliceHeader.PatientBirthDate = '';
    sliceHeader.ProtocolName = '';
    %sliceHeader.PatientPosition = 'HFS';

    


	% Time to save
	slice = squeeze(img(:,:,j));
	dicomwrite(int16(slice),fullfile(outputFolder, sliceName),sliceHeader);
		
        try
		%waitbar(j/size(img,3),saveBar);
        end
    end


try
	%close(saveBar);
end
aScan.save;
