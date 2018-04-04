classdef model < handle

properties

	

end

properties(SetAccess = protected)

	folder % Folder where model parameters and residual will be saved
	%refScan % Number of reference scan
    nScans % Number of scans
	comment % (optional) Descripton for this model
	runScans % Array of scan numbers from the referred study which are included in this model
	uuid % Universal unique identifier
    study % Reference to study object referenced by this model
	registration % Reference to registration object  
    sequence % 4DCT sequences generated using this model
    residualStatistics % Residual in the lung region

end

properties(Access = protected)

end

methods
	

    %% Constructor
	function aModel = model(aStudy, aRegistration, runScans)
	 
    % Set registration and study references
	aModel.study = aStudy;
    aModel.registration = aRegistration;
    
    % Set uuid
    aModel.uuid = char(java.util.UUID.randomUUID);

	% Set run scans.  If none specified, run all scans
    if (exist('runScans','var'))
        runScans = sort(runScans,'ascend');
        aModel.nScans = numel(runScans);
        aModel.runScans = runScans(:)';
        
    else
    	aModel.nScans = aStudy.nScans;
        aModel.runScans = 1:aStudy.nScans;
    end
    
    chkmkdir(fullfile(aStudy.folder,'models'));
	chkmkdir(fullfile(aStudy.folder,'models',aModel.uuid));
    aModel.folder = fullfile(aStudy.folder,'models',aModel.uuid);
    
    end
    
    %% Load parameters
    
    function [aX,aY,aZ] = get_alpha(aModel)
    [aX,aY,aZ] = load_parameters(aModel,'alpha');
    end

    function [bX,bY,bZ] = get_beta(aModel)
    [bX,bY,bZ] = load_parameters(aModel,'beta');
    end
        
    function [cX,cY,cZ] = get_constant(aModel)
    [cX,cY,cZ] = load_parameters(aModel,'constant');
    end
     
    %% get/set
	function aModel = set(aModel,property,value)
	aModel.(property) = value;
	end

	function value = get(aModel,property)
	value = aModel.(property);
    end


end

methods (Static)
    
    %debug = fit_gpu(v,f,dim,startSlice,dRegistration)
    %debug = load_parameters(aModel,p);
    [iX,iY,iZ] = invert_deformation_field_gpu(dX,dY,dZ,nIterations)
    [parameters, modelFit] = fit_gpu(v,f,dim,startSlice, dRegistration)

    
end

events
end

end



