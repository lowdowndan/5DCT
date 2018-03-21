classdef patient < handle

properties

end
	
properties(SetAccess = protected)	
    
    id % Identification number for this patient (MRN). 	
	study % Array of study objects associated with this patient
	first % First name
	last % Last name
    %mrn % Medical record number
	filename % Filename of the .mat file in which the object is saved.  Default: "patient.mat"
end	

properties(Access = protected)
	statusListener
end	

methods
	
    function aPatient = patient(id) 
	
    %% Was an id given?
    
    if(~exist('id','var'))
    
    % Get id and name
    userResp = inputdlg({'MRN','First', 'Last'}, 'Enter patient information.', [1 40]);
    id = str2double(userResp{1});
    first = userResp{2};
    last = userResp{3}; 

    aPatient.first = first;
    aPatient.last = last;
        
        
    else
        
    % aPatient = patient(id) creates a patient object with identification number id.
    	
    %% Check if this is test/qa patient	
	if strcmp(id,'test')
	folder = fullfile(getDataDir,'test','testPatient');
	chkmkdir(folder);
	aPatient.filename = fullfile(folder,'patient.mat');
	aPatient.id = 'test';
	return;
	end

    %% Validate input
    validateattributes(id,{'numeric'},{'nonnegative','real','finite','nonnan','<=',9999999,'numel',1});
	
    end

    aPatient.id = id;

	% Create patient folder
	folder = fullfile(getDataDir,sprintf('%07d',aPatient.id));
	chkmkdir(folder);
    
    % Create model and study folders
    %chkmkdir(fullfile(folder,'studies'));
   % chkmkdir(fullfile(folder,'models'));

	aPatient.filename = fullfile(folder,'patient.mat');

    	if exist(aPatient.filename)
    	    error('Patient already exists.')
        end

    end
    
    function save(aPatient)
    save(aPatient.filename,'aPatient');
    save([aPatient.filename '.bak'],'aPatient');
    end
	
    function update(aPatient,eventSrc,eventData)
    aPatient.save;
    end

	function obj = saveobj(obj)
	obj.statusListener = [];
	end

	function aPatient = set(aPatient,property,value)
	aPatient.(property) = value;
	end

	function value = get(aPatient,property)
	value = aPatient.(property);
	end
end

methods(Static)

    %function obj = loadobj(obj)
   % obj.statusListener = addlistener(obj,'statusChange',@obj.update);
   % end

end

events
        statusChange
end
end








