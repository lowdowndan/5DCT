classdef registration < handle

properties



end

properties(SetAccess = protected)

	folder % Folder where registration results will be saved
	refScan % Number of reference scan
	comment % (optional) Descripton for this model
	uuid % Universal unique identifier
	    
	patient % Reference to patient object which contains this model
	study % Reference to study object referenced by this model
	
	algorithm % Registration algorithm used
	parameters % Parameters used for
	sliceFolder % Folder containing sliced DVFs
    corSlice % Representative coronal slice
    sagSliceL % Representative left sagittal slice
    sagSliceR % Representative right sagittal slice


end

properties(Access = protected)

end

methods
	

	function aRegistration = registration(aStudy, refScan)
	 
	% Se study reference
	aRegistration.study = aStudy;
    
	% Set uuid
	aRegistration.uuid = char(java.util.UUID.randomUUID);

    % Set registration folder
	chkmkdir(fullfile(aStudy.folder,'registrations'));
    aRegistration.folder = fullfile(aStudy.folder,'registrations',aRegistration.uuid);
	chkmkdir(aRegistration.folder);
	
	% Set reference scan
	if(~exist('refScan','var'))

	aRegistration.refScan = 1;
	else
	aRegistration.refScan = refScan;
    end
  
	% Set algorithm
	aRegistration.algorithm = 'deeds';

	% Set parameters
	aRegistration.parameters = struct('alpha',2.0,'samples',128);

	% Slice folder
	aRegistration.sliceFolder = fullfile(aRegistration.folder,sprintf('sliced_%02d',refScan));
	chkmkdir(aRegistration.sliceFolder);

	% Record what scan is the reference
	refScanCmd = ['touch "' fullfile(aRegistration.folder,sprintf('ref_%02d.txt',refScan)) '"'];
	system(refScanCmd);
    end

     
	function aRegistration = set(aRegistration,property,value)
	aRegistration.(property) = value;
	end

	function value = get(aRegistration,property)
	value = aRegistration.(property);
	end
end

methods (Static)
DVF = import(baseFilename)
end

events
end

end



