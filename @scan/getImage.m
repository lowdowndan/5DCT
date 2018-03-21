function img = getImage(aScan)
% img = aScan.getImage loads the image data stored at aScan.niiFilename.
img = load_nii(aScan.niiFilename);
img = img.img;
end
