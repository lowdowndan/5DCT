%% generate_image

function img = generate_image(aModel,v,f, aX, aY, aZ, bX, bY, bZ, cX, cY, cZ, img)

% Load parameters if they are not provided as input arguments
if(~exist('aX','var'))
    
    [aX, aY, aZ] = aModel.get_alpha;
    [bX, bY, bZ] = aModel.get_beta;
    [cX, cY, cZ] = aModel.get_constant;
    
end


if(~exist('img','var'))
    img = aModel.registration.get_average_image;
end
    

[dX, dY, dZ] = aModel.get_deformation(v,f,aX,aY,aZ,bX,bY,bZ,cX,cY,cZ);
[X,Y,Z] = meshgrid(1:size(img,1),1:size(img,2),1:size(img,3));
img = interp3(X,Y,Z,img, X + dX, Y + dY, Z + dZ, 'linear', -1024);