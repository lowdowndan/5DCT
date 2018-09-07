%WRITE_DICOM_ANON Write scan to anonymized DICOM.
% aScan.write_dicom_anon(outputFolder) generates a random patient name, ID, and 
% StudyUID and writes the scan to outputFolder.
%
% aScan.write_dicom_anon(outputFolder, first, last, id, studyUID, seriesDescription) writes the
% scan to outputFolder with DICOM tags PatientName.GivenName = First,
% PatientName.FamilyName = last, PatientID = id, StudyUID = studyUID,
% and SeriesDescription = SeriesDescription.

function aScan = write_dicom_anon(aScan, outputFolder, first, last, id, studyUID, seriesDescription)


%% Create output folder
chkmkdir(outputFolder);

% Anonymize
anonStr = randi(9,1,6);
anonStr = num2str(anonStr);
anonStr = anonStr(~isspace(anonStr));

% First
if ~(exist('first','var') && ~isempty(first))

	first = ['Anon' anonStr];
end

% Last
if ~(exist('last','var') && ~isempty(last))

	last = ['Anon' anonStr];
end

% ID
if ~(exist('id','var') && ~isempty(id))

	id = ['Anon' anonStr];
end

% studyUID
if ~(exist('studyUID','var') && ~isempty(studyUID))

	studyUID = dicomuid;
end

% seriesDescription
if ~(exist('seriesDescription','var') && ~isempty(seriesDescription))

	if(isempty(aScan.seriesDescription))
	seriesDescription = 'Anonymized 5D';
	else
	seriesDescription = aScan.seriesDescription;
	end
end


% study description
if isempty(aScan.studyDescription)
    aScan.studyDescription = sprintf('5D Clinical Protocol');
%    warning('No StudyDescription tag for this scan.  Writing default.')
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

%% Stuff to scrub from header
% from:
% Aryanto KY, Oudkerk M, van Ooijen PM. Free DICOM de-identification tools in
% clinical research: functioning and safety of patient privacy. European
% radiology. 2015 Dec 1;25(12):3685-95.

scrubFields = {
'StudyDate'
'SeriesDate'
'AcquisitionDate'
'ContentDate'
'OverlayDate'
'CurveDate'
'AcquisitionDatetime'
'StudyTime'
'SeriesTime'
'AcquisitionTime'
'ContentTime'
'OverlayTime'
'CurveTime'
'AccessionNumber'
'InstitutionName'
'InstitutionAddress'
'ReferringPhysiciansName'
'ReferringPhysiciansAddress'
'ReferringPhysiciansTelephoneNumber'
'ReferringPhysicianIDSequence'
'InstitutionalDepartmentName'
'PhysicianOfRecord'
'PhysicianOfRecordIDSequence'
'PerformingPhysiciansName'
'PerformingPhysicianIDSequence'
'NameOfPhysicianReadingStudy'
'PhysicianReadingStudyIDSequence'
'OperatorsName'
'PatientsName'
'PatientID'
'IssuerOfPatientID'
'PatientsBirthDate'
'PatientsBirthTime'
'PatientsSex'
'OtherPatientIDs'
'OtherPatientNames'
'PatientsBirthName'
'PatientsAge'
'PatientsAddress'
'PatientsMothersBirthName'
'CountryOfResidence'
'RegionOfResidence'
'PatientsTelephoneNumbers'
'StudyID'
'CurrentPatientLocation'
'PatientsInstitutionResidence'
'DateTime'
'Date'
'Time'
'PersonName'};


% Remove
sliceFields = fieldnames(sliceHeader);
matches = ismember(sliceFields,scrubFields);
nScrub = nnz(matches);
toScrub = find(matches);

for iScrub = 1:nScrub
	sliceHeader.(sliceFields{toScrub(iScrub)}) = '';
end


% Reinsert anonymized tags 
sliceHeader.PatientID = id;
sliceHeader.PatientName = struct('FamilyName',last,'GivenName',first);
sliceHeader.StudyInstanceUID = studyUID;
sliceHeader.SeriesDescription = seriesDescription;


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

