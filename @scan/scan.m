classdef scan < handle

properties
	filename % Filename of scan data structure
	img % Image data
	dim % Dimensions of image data
	elementSpacing % x, y, and z element spacing (center to center)
	v  % Breathing amplitude measurements 
	f  % Breathing rate measurements 
	ekg  % Ekg rate measurements 
	t % Scan times 
	direction % Boolean.  True if scan was acquired caudocranially, false if craniocaudal
    seriesDescription % Description for this scan
    studyDescription % Description for this study
end

properties(SetAccess = protected)
	dicoms % Cell array of filenames for the dicom slices comprising this image
	original % Boolean.  True if this scan was actually acquired, false if it is a model derived image
	zPositions % Slice positions
	niiFilename % Filename of image data in .nii format
	number % Order of this scan in the original acquisition
    imagePositionPatient % DICOM RCS coordinates of the center of the first voxel
    seriesUID % Dicom UID for this image
	studyUID % Dicom UID for this study
    parent % Reference to object that created this scan
end

properties(Access = protected)
	originalDim % Dimensions before cropping
	originalElementSpacing % Element spacing before resampling to 1x1x1
    originalZPositions % Z positions before resampling to 1x1x1.
    originalImagePositionPatient % Original ImagePositionPatient value
    acquisitionInfo % DICOM header information
  end

methods

    function aScan = scan(parent, acquisitionInfo, imagePositionPatient, zPositions)

    aScan.original = false;
	aScan.direction = 1;
    aScan.seriesUID = dicomuid;
    aScan.acquisitionInfo = acquisitionInfo;
    aScan.imagePositionPatient = imagePositionPatient;
    aScan.zPositions = zPositions;
    aScan.parent = parent;
	end

	function aScan = save(aScan)
	% aScan.save saves this scan object as a .mat file with filename aScan.filename.
	if isempty(aScan.filename)
		%aScan.filename = uiputfile('*.mat','Save as');
        aScan.filename = fullfile(aScan.parentFolder,aScan.seriesUID);
        warning(sprintf('No filename specified. Saving to %s.\n', aScan.filename));
	end

	save(aScan.filename,'aScan');
    end

    function aScan = set(aScan,property,value)
    % aScan.set('propertyName', value) sets property 'propertyName' to value.
	aScan.(property) = value;
	end


	function value = get(aScan,propertyName);
	% value = aScan.get('propertyName') returns the value of property 'propertyName'.
	value = aScan.(propertyName);
	end


end

methods (Static)
push(scanFolder)
end

end
