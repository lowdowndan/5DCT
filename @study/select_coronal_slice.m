%% select_coronal_slice

function select_coronal_slice(aStudy, img)

% Display reference image
vi(img)

validEntry = false;

while(~validEntry)
    
coronalSlice = inputdlg('Enter a coronal slice number.  2D QA plots will be displayed at that slice.', 'Representative slice', 1)
coronalSlice = coronalSlice{1};

if(~isempty(coronalSlice))
    coronalSlice = str2num(coronalSlice);
    
    if(isnumeric(coronalSlice))
        if(coronalSlice > 0 && coronalSlice < size(img,1)
    
% Validate




