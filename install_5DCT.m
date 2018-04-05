%% Installation script for 5DCT toolbox

%% Make sure system requirements are satisfied.

% Check for toolboxes
imgToolbox = ~isempty(ver('images'));
if (~imgToolbox)
	error('Image processing toolbox required.');
end


statToolbox = ~isempty(ver('stats'));
if (~statToolbox)
	error('Statistics toolbox requried.');
end

parToolbox = ~isempty(ver('distcomp'));
if (~parToolbox)
	error('Parallel computing toolbox required.');
end

% Query GPU if parallel computing toolbox is available
gpu = gpuDevice;
assert(isa(gpu,'parallel.gpu.CUDADevice'), 'CUDA capable GPU required.');



% Throw warning if no image processing toolbox is found
if ~systemParams.imgToolbox
warning('Image processing toolbox is required to support DICOM import/export.');
end
%% Set data directory


%% Check if toolbox is already installed
if (ispref('fiveD','dataDir') && ispref('fiveD','installDir')) 

% Toolbox preferences found. Overwrite?

	overwrite = questdlg('Toolbox already installed.  Select new install directories?', '5D Toolbox', 'Yes', 'No','No');

	if(strcmp(overwrite,'No'))

	else

	% Prompt user
	dataDir = uigetdir('/','Select a directory to store patient data.');
	% Set
	setpref('fiveD','dataDir',dataDir);
	% Set install directory
	installDir = fileparts(mfilename('fullpath'));
	setpref('fiveD','installDir',installDir);

	end
else
% No toolbox preferences found.  Prompt

	% Prompt user
	dataDir = uigetdir('/','Select a directory to store patient data.');
	% Set
	setpref('fiveD','dataDir',dataDir);
	% Set install directory
	installDir = fileparts(mfilename('fullpath'));
	setpref('fiveD','installDir',installDir);

end


%% Add install directory to path
disp('Adding 5DCT Toolbox folders to path.');
addpath(genpath(installDir));
savepath;


%% Compile mex functions
ogDir = pwd;
cd(fullfile(fiveDpath,'@model'))

% fit_gpu
mexCmd = ['mexcuda -I' fullfile(fiveDpath,'include') ' -L/usr/local/cuda/lib64 -lcublas ' 'fit_gpu.cu'];
eval(mexCmd);

% invert_deformation_field_gpu
mexCmd = ['mexcuda -I' fullfile(fiveDpath,'include') ' invert_deformation_field_gpu.cu'];
eval(mexCmd);

% memQuery
cd(fullfile(fiveDpath,'supportFunctions'));
mexCmd = ['mexcuda -I' fullfile(fiveDpath,'supportFunctions') ' memQuery.cu'];
eval(mexCmd);

% cudaReset
mexCmd = ['mexcuda -I' fullfile(fiveDpath,'supportFunctions') ' cudaReset.cu'];
eval(mexCmd);


% Return to original directory
cd(ogDir);




msg = ['Please add ' fullfile(installDir,'supportFunctions') ' to the system path.']; 
disp(msg);



