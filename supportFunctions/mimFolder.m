function mimFolder = mimFolder(grandParentFolder)

parentFolder = dir(grandParentFolder);
isDir = [parentFolder.isdir];
parentFolder = {parentFolder.name};
parentFolder = parentFolder(isDir);
parentFolder = setdiff(parentFolder, {'.','..'});
parentFolder = parentFolder{1};

mimFolder = dir(fullfile(grandParentFolder, parentFolder));
isDir = [mimFolder.isdir];
mimFolder = {mimFolder.name};
mimFolder = mimFolder(isDir);
mimFolder = setdiff(mimFolder, {'.','..'});
mimFolder = mimFolder{1};
mimFolder = fullfile(grandParentFolder,parentFolder,mimFolder);
end
