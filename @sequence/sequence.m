classdef sequence < handle

properties

  
end

properties(SetAccess = protected)

	folder % Folder where model parameters and residual will be saved
	comment % (optional) Descripton for this model
	uuid % Universal unique identifier
    
    model
    breath

    reconstructionPoints
    exhaleAmplitudes % Amplitudes, expressed in %, to reconstruct images at (exhale)
    inhaleAmplitudes % Amplitudes, expressed in %, to reconstruct images at (inhale)

    studyUID % DICOM UID for the images in this sequence
    scans
    dicomFolders


end

properties(Access = protected)

end

methods
	

    %% Constructor
	function aSequence = sequence(aModel, aBreath)
	 
    % References
    aSequence.model = aModel;
    aSequence.breath = aBreath;
    
    % Structure containing information for each image
    aSequence.reconstructionPoints = struct('v',[],'f',[],'state',[],'amplitude',[],'description',[],'seriesUID',[]);
    
    % Mimic Siemens 8-phase format
    aSequence.exhaleAmplitudes = [0 25 50 75 100];
    aSequence.inhaleAmplitudes = [25 50 75];
    
    aSequence.uuid = char(java.util.UUID.randomUUID);
    
    chkmkdir(fullfile(aSequence.model.folder,'sequences'));
    aSequence.folder = fullfile(aSequence.model.folder,'sequences',aSequence.uuid);
    chkmkdir(aSequence.folder);
    aSequence.studyUID = dicomuid;
    
    end
  
   
    
    %% get/set
	function aSequence = set(aSequence,property,value)
	aSequence.(property) = value;
	end

	function value = get(aSequence,property)
	value = aSequence.(property);
    end


end

methods (Static)

end

events
end

end



