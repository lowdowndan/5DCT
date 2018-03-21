function writeNii(aScan, filename)
% aScan.writeNii writes the image data stored in aScan.img to .nii format using the provided filename.

% Check for .nii extension in filename, add if not present
if length(filename) < 4

	if ~strcmp(lower(filename(end - 3: end)), '.nii')
	filename = [filename '.nii'];
	end
else
	filename = [filename '.nii'];
end



% Create and save .nii img
imgNii = make_nii(aScan.img,aScan.elementSpacing);
save_nii(imgNii,filename);
aScan.niiFilename = filename;



