function DVF = import_registration(baseFilename)

% Read components
uNii = load_nii([baseFilename '_flowu.nii']);
vNii = load_nii([baseFilename '_flowv.nii']);
wNii = load_nii([baseFilename '_floww.nii']);

DVF = zeros([size(uNii.img,1) size(uNii.img,2), size(uNii.img,3) 3], 'single');
DVF(:,:,:,1) = uNii.img;
DVF(:,:,:,2) = vNii.img;
DVF(:,:,:,3) = wNii.img;
