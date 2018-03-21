%% configureSystem	Set up 5D toolbox
% 
% 

function systemParams = configureSystem


if exist(fullfile(prefdir,'fiveD_systemParams.dat'))
overwriteParams = questdlg('Configuration file already exists. Would you like to reconfigure?','Yes','No');	
	switch overwriteParams
	case 'Yes'
	case 'No'
		systemParams = getSystemParams;
		return;
	case 'Cancel'
	
		systemParams = getSystemParams;
		return;
	end
end

systemParams = struct;

% Check for toolboxes
systemParams.imgToolbox = ~isempty(ver('images'));
systemParams.statToolbox = ~isempty(ver('stats'));
systemParams.parToolbox = ~isempty(ver('distcomp'));

% Query GPU if parallel computing toolbox is available
if systemParams.parToolbox
	
	if gpuDeviceCount
		gpu = struct;
		gpudev = gpuDevice;
		gpu.mem = gpudev.TotalMemory;
		gpu.compute = str2num(gpudev.ComputeCapability);
		systemParams.gpu = gpu;
	else
		systemParams.gpu = 0; 
	end

else
	systemParams.gpu = 0;
end


% Throw warning if no image processing toolbox is found
if ~systemParams.imgToolbox
warning('Image processing toolbox is required to support DICOM import/export.');
end

% Get data directory
selectMsg = 'Choose a location to save 5DCT patient data.';
disp(selectMsg)
dataDir = uigetdir('',selectMsg);

if ~dataDir
	error('No directory was selected.');
end

dataDir = fullfile(dataDir,'5DCT_data');
mkdir(dataDir);
systemParams.dataDir = dataDir;

% Make test/qa directory
mkdir(fullfile(dataDir,'test'));
% Get path
systemParams.path = prefdir;

% Write systemParams
save(fullfile(systemParams.path,'fiveD_systemParams.dat'),'systemParams');


