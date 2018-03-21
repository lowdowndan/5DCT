function sliceImg = getSlice(image3d,dim,slice);
% Get slice of three dimensional volume
% Returns slice number slice in dimension dim of image3d.
% Removes singleton dimensions and rotates the image 90 degrees

switch dim
case 1
	sliceImg = squeeze(image3d(slice,:,:));
case 2
	sliceImg = squeeze(image3d(:,slice,:));
case 3
	sliceImg = squeeze(image3d(:,:,slice));
otherwise
	error('dim must be 1, 2, or 3');
end

sliceImg = imrotate(sliceImg,90);


