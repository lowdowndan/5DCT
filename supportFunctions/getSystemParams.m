%% getSystemParams	Returns a structure containing system parameters
%determined during initial setup of the 5DCT Toolbox.
%
%
% TODO:
% Write to prefdir path
function systemParams = getSystemParams

try
systemParams = load(fullfile(prefdir,'fiveD_systemParams.dat'),'-mat');
systemParams = systemParams.systemParams;
catch me
	if strcmp(me.identifier, 'MATLAB:load:couldNotReadFile')'
		cause = MException('FiveDToolbox:noList','No patient list found.');
		me = addCause(me,cause);
		warning('No system parameters file found. Configuring... ');
		configureSystem;

	else
	rethrow(me);
	end
end

systemParams = load(fullfile(prefdir,'fiveD_systemParams.dat'),'-mat');
systemParams = systemParams.systemParams;


end
