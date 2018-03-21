%% getDataDir	Returns the path to folder where 5DCT Toolbox patient data is stored.
%
%
function dataDir = getDataDir;

systemParams = getSystemParams;
dataDir = systemParams.dataDir;
end


