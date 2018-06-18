%% Script to add new 5DCT patient

% Get info
valid = false;
while (~valid)
userResp = inputdlg({'Patient ID (MRN)','Given (First) Name', 'Family (Last) Name'}, 'Enter patient information.', [1 50]);

% Was ID number provided?

if(isempty(userResp) || strcmp(userResp{1},''))
    waitfor(errordlg('Enter patient ID number.','Error', 'modal'));
else
    
% Is ID number valid?
id = str2double(userResp{1});
validateattributes(id,{'numeric'},{'nonnegative','real','finite','nonnan','<=',999999999999,'numel',1});

% Set first and last, okay if they're empty
first = userResp{2};
last = userResp{3};

valid = true;
end

end

aPatient = patient(id);
aPatient.set('first',first);
aPatient.set('last',last);

workflow_fived(aPatient);
