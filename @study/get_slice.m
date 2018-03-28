function slice = get_slice(img,dim,slice)
% Get slice of three dimensional volume
% Returns slice number slice in dimension dim of image3d.
% Removes singleton dimensions and rotates the image 90 degrees

switch dim
case 1
	slice = squeeze(img(slice,:,:));
case 2
	slice = squeeze(img(:,slice,:));
case 3
	slice = squeeze(img(:,:,slice));
otherwise
	error('dim must be 1, 2, or 3');
end

slice = imrotate(slice,90);
%slice = squeeze(slice);


