%% Set the starting paths for convenience
function set_5DCT_working_directories

if (ispref('fiveD','studyStartDir'))
% Toolbox preference found. Overwrite?
userResp = questdlg('Change existing new study start directory?', '5D Toolbox', 'Yes', 'No','No');

    if(strcmp(userResp,'Yes'))
    setDir = true;
    else
    setDir = false;
    end
    
else
    setDir = false;
end


if(setDir)
    startDir = uigetdir('','Select starting directory for new study dialogue.');
    setpref('fiveD','studyStartDir',startDir);
end

