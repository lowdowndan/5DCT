%% Load image (deeds)

function img = get_image(aRegistration, imgNumber)

img = load_nii(fullfile(aRegistration.folder,sprintf('%02d_deformed.nii',imgNumber)));
img = img.img;
end