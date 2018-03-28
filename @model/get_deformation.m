%% get_deformation

function [dX, dY, dZ] = get_deformation(aModel, v,f, aX, aY, aZ, bX, bY, bZ, cX, cY, cZ)

% Load parameters if they are not provided as input arguments
if(~exist('aX','var'))
    
    [aX, aY, aZ] = aModel.get_alpha;
    [bX, bY, bZ] = aModel.get_beta;
    [cX, cY, cZ] = aModel.get_constant;
    
end

% Accept scalar v,f values only
if (numel(v) ~= 1) || (numel(f) ~= 1)
   % error('v,f must be scalars.');
     
   
    assert(length(v) == size(aX,3), 'If v, f are vector-valued they must have length equal to the number of slices');

    iX = zeros(size(aX),'single');
    iY = zeros(size(aX),'single');
    iZ = zeros(size(aX),'single');

    for iSlice = 1:length(v)
    
    iX(:,:,iSlice) = cX(:,:,iSlice) + (aX(:,:,iSlice) .* v(iSlice)) + (bX(:,:,iSlice) .* f(iSlice));
    iY(:,:,iSlice) = cY(:,:,iSlice) + (aY(:,:,iSlice) .* v(iSlice)) + (bY(:,:,iSlice) .* f(iSlice));
    iZ(:,:,iSlice) = cZ(:,:,iSlice) + (aZ(:,:,iSlice) .* v(iSlice)) + (bZ(:,:,iSlice) .* f(iSlice));
    
    end
   
   
else
    
% Calculate forward deformations
iX = cX + (aX .* v) + (bX .* f);
iY = cY + (aY .* v) + (bY .* f);
iZ = cZ + (aZ .* v) + (bZ .* f);
    
end


iX = single(iX);
iY = single(iY);
iZ = single(iZ);
    
% Invert deformation
[dX,dY,dZ] = model.invert_deformation_field_gpu(iX,iY,iZ,20);
