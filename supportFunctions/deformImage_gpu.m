function imgOut = deformImage_gpu(imgIn, dX, dY, dZ)
   
[Xi, Yi, Zi] = meshgrid(1:size(imgIn,2),1:size(imgIn,1),1:size(imgIn,3));

Xi = Xi + dX;
Yi = Yi + dY;
Zi = Zi + dZ;

Xi = single(Xi);
Yi = single(Yi);
Zi = single(Zi);
	
% GPU Interpolation
imgIn = single(imgIn);
imgOut = trilinterp(imgIn,Xi,Yi,Zi);
    
end
