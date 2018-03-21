%% chkmkdir: Create a directory if it doesn't already exist
function chkmkdir(dirPath)

if ~ exist(dirPath,'dir')
	mkdir(dirPath)
end

