classdef study < handle

properties
    comment % Optional descripton for this study


end
	
properties(SetAccess = protected)

    date % Date and time of study
	dicomFolder % Folder containing dicoms for free breathing scans
	data % Drift-compensated bellows, ekg and x-ray on data from daq

	channels % Structure with column indices into the LabVIEW data matrix for time, bellows voltage, x-ray on and ekg.
	bellowsInhaleDirection % 1 if inhalation is positive, 0 if inhalation is negative.  If 0, trace was flipped. 
    bellowsSmoothingWindow % Window used for Savitzky-Golay smoothing of bellows signal
    startScan % Row indices that mark the beginnings of scans in data matrix
	stopScan % Row indices that mark the ends of scans in data matrix

	nScans % Number of scans in study
	scans % Scan objects

	v % Synchronized breathing amplitude measurements for all slices
	f % Synchronized breathing rate measurements for all slices
	ekg % Synchronized ekg measurements for all slices
	drift % Drift rate of the bellows (V/.01s)
    direction % Direction of scan.  1 is foot to head, 0 head to foot.
    dim % Dimensions of images

    abdomenHeights
    calibrationVoltages
    calibrationTimes
    
    initialCorrelation
    driftedCorrelation
    scanRange % Minimum and maximum couch positions common to all scans in the study
    corSlice % Representative coronal slice, used for QA report images
    patient % Reference to patient data structure which contains this study
	rawData % Data acquired during the scan
	folder % Folder where data is stored
	sampleRate % Sample rate of the data acquisition
    dataRange % Relevant region of the breathing trace
    status % Status of the study object
    scanIDs % DICOM UIDs of the scans included in this study
    importedScanIDs % New UIDs for the scans after they have been imported
    sliceCounts % Number of slices in each included scan
    studyUID % DICOM UID for the resampled original scans.  Distinct from the UID of the original study.
    uuid % Universal unique identifier for this study object.  Used as folder name.  This is specific to the 5D Toolbox and is not a DICOM Study UID.
    acquisitionInfo % Set of DICOM header information for output
    zPositions % Slice locations for this study's geometry
    imagePositionPatient % DICOM RCS coordinates of the first voxel center.
    registration % Registrations associated with this study
    
    breath % Representative breaths
    model % 5D Model

end



methods
	
	function aStudy = study(aPatient, bellowsDataFilename, dicomFolder, nScans)

	aStudy.patient = aPatient;

	% Set properties necessary to sync
	aStudy.dicomFolder = dicomFolder;
	aStudy.rawData = importdata(bellowsDataFilename);
    aStudy.data = aStudy.rawData;
	aStudy.nScans = nScans;
    
	 % Generate uuid
	 aStudy.uuid = char(java.util.UUID.randomUUID);
    
	% Make study folder, use uuid as folder name
	chkmkdir(fullfile(fileparts(aPatient.filename),'studies'));
	aStudy.folder = fullfile(fileparts(aPatient.filename),'studies', aStudy.uuid);
	chkmkdir(aStudy.folder);
	chkmkdir(fullfile(aStudy.folder,'documents'));
   
    % Save
	notify(aStudy.patient,'statusChange');

    end
    
    function aScan = getScan(aStudy, scanNumber)
        aScan = load(fullfile(aStudy.folder, sprintf('%02d.mat', scanNumber)));
        aScan = aScan.aScan;
    end

    
    function aStudy = set(aStudy,property,value)
	aStudy.(property) = value;
	end

	function value = get(aStudy,property)
	value = aStudy.(property);
	end

end

methods (Static)
    flow = get_flow(v,sampleRate)
    vSmooth = smooth(v)
    slice = get_slice(img, dim, slice)
end

events
end
end



