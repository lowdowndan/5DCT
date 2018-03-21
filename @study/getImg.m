%% getImg

function img = getImg(aStudy, scanNumber)

img = load_nii(fullfile(aStudy.folder,'nii', sprintf('%02d.nii', scanNumber)));
img = img.img;
end