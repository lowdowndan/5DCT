%% get_mask
function mask = get_mask(aStudy, scanNo)

if(~exist('scanNo','var'))
    scanNo = 1;
    warning('No scan number specified.  Returning mask for scan 01.');
end


try
load(fullfile(aStudy.folder,sprintf('mask_%02d.mat',scanNo)));
catch ME
    
    if(strcmp('MATLAB:load:couldNotReadFile', ME.identifier))
       disp('No lung mask found. Generating mask using Pulmonary Toolkit.')
        mask = aStudy.ptk_mask(scanNo);
        % error('No mask for scan %02d found.  Please use import_mask method to import a lung segmentation from DICOM, or ptk_mask method to automatically generate a lung segmentation using the pulmonary toolkit (ptk).',scanNo)
    else
        error('Unable to load mask.');
    end
end