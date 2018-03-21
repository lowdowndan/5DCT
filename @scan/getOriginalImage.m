function img = getOriginalImage(aScan, img)
% img = aScan.getOriginalImage returns the image associated with this scan in its original size (i.e. 512 x 512 x nSlices, rather than the typical 500 x 500 x nSlices after resampling to 1x1x1)

elementSpacing = aScan.originalElementSpacing;


if nargin < 2
    
    if isempty(aScan.img)
    img = aScan.getImage;
    else
    img = aScan.img;
    end

end


dx = elementSpacing(1);
dy = elementSpacing(2);
dz = elementSpacing(3);
img = resampleImage(img, dx, dy, dz, aScan.direction);

% Fix off-by-one image sizes due to rounding errors after resampling
 if size(img,1) < 512 || size(img,2) < 512
     warning('image is incorrect size; fixing off-by-one');
 imgFull = ones([512, 512, aScan.originalDim(3)]) * min(img(:));
 imgFull(1:size(img,1),1:size(img,2),1:size(img,3)) = img;
 img = imgFull;
 
 elseif size(img,1) > 512
 img = img(1:512,1:512,:);
 end
 end




