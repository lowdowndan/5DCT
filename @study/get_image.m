%% get_mask
function img = get_image(aStudy, imgNumber)

img = load_nii(fullfile(aStudy.folder,'nii',sprintf('%02d.nii',imgNumber)));
img = img.img;