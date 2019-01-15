%% Set the starting paths for convenience
function set_5DCT_working_directories

if (ispref('fiveD','studyImgStartDir'))
% Toolbox preference found. Overwrite?
userResp = questdlg('Change existing new study image start directory?', '5D Toolbox', 'Yes', 'No','No');

    if(strcmp(userResp,'Yes'))
    setDir = true;
    else
    setDir = false;
    end
    
else
    setDir = true;
end


if(setDir)
    startDir = uigetdir('','Select a directory to look for CT images in the new study dialogue.');
    setpref('fiveD','studyImgStartDir',startDir);
end

if (ispref('fiveD','studyBellowsStartDir'))
% Toolbox preference found. Overwrite?
userResp = questdlg('Change existing new study bellows start directory?', '5D Toolbox', 'Yes', 'No','No');

    if(strcmp(userResp,'Yes'))
    setDir = true;
    else
    setDir = false;
    end
    
else
    setDir = true;
end


if(setDir)
    startDir = uigetdir('','Select a directory to look for bellows data files in the new study dialogue.');
    setpref('fiveD','studyBellowsStartDir',startDir);
end

