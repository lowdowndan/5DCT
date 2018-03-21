%% getReconError: Get the original scan reconstruction error.
% Calculates the magnitude of deformation vector fields written
% by deform_images_psuedo_originalscans_toolbox after registration
% of model generated pseudoscans to originally acquired scans. 
% These magnitudes are then deformed to the reference image 
% geometry.
%
% Arguments (in)
% patient: patient data structure from 5D Toolbox

%% TODO:
% create registration folder property for model


function getReconError(aModel)

aStudy = aModel.study;
errorFolder = fullfile(aModel.folder,'reconError');
chkmkdir(errorFolder);
reconBar = waitbar(0,'Calculating original scan reconstruction error and deforming to reference image geometry...');

for iScan = 1:aStudy.nScans

   	% Load error map
    scanFolder = fullfile(aModel.folder,'original',sprintf('%02d',iScan));
    errorMap = readDeedsFlow(fullfile(scanFolder,'deedsOut'));
	
	% Take magnitude of deformation vectors as voxel error
	errorMap = bsxfun(@hypot, squeeze(errorMap(:,:,1,:)), bsxfun(@hypot, squeeze(errorMap(:,:,2,:)),squeeze(errorMap(:,:,3,:))));
    errorMap = single(errorMap);
    
    %if iScan ~= 1
	% Load deformation field going to reference geometry
	refDVF = readDeedsFlow(fullfile(aModel.registrationFolder, sprintf('%02d', iScan)));
    refDVF = single(refDVF);

    % Deform image to reference geometry
	errorMap = deformImage_gpu(errorMap, squeeze(refDVF(:,:,1,:)), squeeze(refDVF(:,:,2,:)), squeeze(refDVF(:,:,3,:)));
    %end
    
	% Save error map
	
	errorNii = make_nii(errorMap);
	save_nii(errorNii,fullfile(errorFolder,sprintf('%02d.nii',iScan)));

	try
		waitbar(iScan/aStudy.nScans,reconBar);
	end

end


try
	close(reconBar);
end


