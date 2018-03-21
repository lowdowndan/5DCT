%% Get average image

function averageImage = get_average_image(aRegistration)

%% Check if average image has already been generated

if exist(fullfile(aRegistration.folder,'averageImage.mat'),'file')
    load(fullfile(aRegistration.folder,'averageImage.mat'));
    return;
end
 
%% Generate average image

averageImage = zeros(aRegistration.study.dim,'single');

for iImage = 1:aRegistration.study.nScans
    
    img = aRegistration.get_image(iImage);
    averageImage = averageImage + img;
    
end

averageImage = averageImage ./ aRegistration.study.nScans;
save(fullfile(aRegistration.folder,'averageImage.mat'),'averageImage');
