%% get_reconstruction
%
% Retreieve reconstruction of original scan

function img = get_reconstruction(aModel, scanNo)

img = load_nii(fullfile(aModel.folder,'original',sprintf('%02d',scanNo),'simulated.nii'));
img = img.img;

end
