%% generate_image

function img = generate_image(aModel,v,f, aX, aY, aZ, bX, bY, bZ, cX, cY, cZ)

% Load parameters if they are not provided as input arguments
if(~exist('aX','var'))
    
    [aX, aY, aZ] = aModel.get_alpha;
    [bX, bY, bZ] = aModel.get_beta;
    [cX, cY, cZ] = aModel.get_constant;
    
end

[dX, dY, dZ] = aModel.get_deformation(v,f,aX,aY,aZ,bX,bY,bZ,cX,cY,cZ);

[X,Y,Z] = meshgrid(1:aModel.study.dim(1),1:aModel.study.dim(2),1:aModel.study.dim(3));

averageImage = aModel.registration.get_average_image;
img = interp3(X,Y,Z,averageImage, X + dX, Y + dY, Z + dZ, 'linear', -1024);