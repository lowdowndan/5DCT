function aModel = fit(aModel)

aStudy = aModel.study;
aRegistration = aModel.registration;         

%% Preallocate

% Residuals
meanResidual = zeros(aStudy.dim, 'single');

%% Scans      
nScans = aModel.nScans;
v = single(aStudy.v);
f = single(aStudy.f);
dim = double(aStudy.dim);

fitBar = waitbar(0,'Fitting model parameters...');

%% Set chunk size

% More than 50 or so results in out of memory error for 12 GB GPU RAM
% Reduced to 30 for clinical workstation 
chunkSize = 30;
nChunks = ceil(aStudy.dim(3) / chunkSize);
nVoxelsSlice = aStudy.dim(1) * aStudy.dim(2);

%% Loop over chunks

%tic
for iChunk = 1:nChunks

% Is this the last chunk?  It will probably have fewer than nSlices in it
if(iChunk < nChunks)

	nSlicesChunk = chunkSize;
else
	nSlicesChunk = aStudy.dim(3) - ((nChunks - 1) * chunkSize);
end


	% Preallocate dRegistration matrix for this chunk (contains dvf data for all voxels all scans)
	dRegistration = zeros(aStudy.nScans, (nVoxelsSlice * 3 * nSlicesChunk),'single');

	% Read data from disk
	for jSlice = 1: nSlicesChunk

	% Slice number
  	iSlice = ((iChunk - 1) * chunkSize) + jSlice;
   
	% What's the first slice of this chunk? (used for deforming surrogate by z vector of DVF)
	if(jSlice == 1)
	startSlice = double(iSlice);
    stopSlice = startSlice + nSlicesChunk - 1;
	end

	% Indices into the dRegistration matrix (each row is 1 scan; columns
	% are X,Y,Z for each voxel.  ie first column is X for voxel 1, second
	% column is Y for voxel 1, and third column Z for voxel 1.

	startInd = ((jSlice - 1) * nVoxelsSlice * 3) + 1;
	stopInd = startInd + (nVoxelsSlice * 3) - 1;

	fSlice = fopen(fullfile(aModel.registration.sliceFolder,sprintf('%04d.dat',iSlice)),'r');

    dVec = fread(fSlice,'single');
	fclose(fSlice);

	% Rearrange
	dVec = reshape(dVec, [aStudy.dim(1) aStudy.dim(2) 3 aStudy.nScans]);
	dVec = permute(dVec,[4 1 2 3]);

	% X
	xVec = dVec(aModel.runScans,:,:,1);
	xVec = xVec(:,:);
	dRegistration(:,startInd + 0:3:stopInd - 2) = xVec;

	% Y
	yVec = dVec(aModel.runScans,:,:,2);
	yVec = yVec(:,:);
	dRegistration(:,startInd + 1:3:stopInd - 1) = yVec;

	% Z
	zVec = dVec(aModel.runScans,:,:,3);
	zVec = zVec(:,:);
    
	dRegistration(:,startInd + 2:3:stopInd - 0) = zVec;
	
	% End loop over slices
    end

    
	% CUDA model fitting
    [parameters, modelFit] = model.fit_gpu(v,f,dim,startSlice, dRegistration);

    xParameters = parameters(:,1:3:end-2);
	yParameters = parameters(:,2:3:end-1);
	zParameters = parameters(:,3:3:end);

	xFit = reshape(modelFit(1,:)', nScans, size(modelFit,2) / nScans);
	yFit = reshape(modelFit(2,:)', nScans, size(modelFit,2) / nScans);
	zFit = reshape(modelFit(3,:)', nScans, size(modelFit,2) / nScans);    

	xResidual = bsxfun(@minus,xFit,dRegistration(:,1:3:end - 2));
	yResidual = bsxfun(@minus,yFit,dRegistration(:,2:3:end - 1));
	zResidual = bsxfun(@minus,zFit,dRegistration(:,3:3:end - 0));
    
    
    % Compute mean residual
    chunkResidual = (bsxfun(@hypot,zResidual,bsxfun(@hypot,xResidual,yResidual)));
    chunkResidual = mean(chunkResidual);
    meanResidual(:,:,startSlice:stopSlice) = reshape(chunkResidual,aStudy.dim(1),aStudy.dim(2),nSlicesChunk);
	%toc

    
    % First chunk?  Create files
    
    if iChunk == 1

		% Delete pre-existing parameters
        if exist(fullfile(aModel.folder,'constantX.dat'),'file')
              
        warning('Overwriting previous parameters.');

        delete(fullfile(aModel.folder,'constantX.dat'));
		delete(fullfile(aModel.folder,'constantY.dat'));
		delete(fullfile(aModel.folder,'constantZ.dat'));
		
        delete(fullfile(aModel.folder,'alphaX.dat'));
		delete(fullfile(aModel.folder,'alphaY.dat'));
		delete(fullfile(aModel.folder,'alphaZ.dat'));
    	
        delete(fullfile(aModel.folder,'betaX.dat'));
		delete(fullfile(aModel.folder,'betaY.dat'));
		delete(fullfile(aModel.folder,'betaZ.dat'));
		end

		% Create files
		fConstantX = fopen(fullfile(aModel.folder,'constantX.dat'),'a');
		fConstantY = fopen(fullfile(aModel.folder,'constantY.dat'),'a');
		fConstantZ = fopen(fullfile(aModel.folder,'constantZ.dat'),'a');

		fAlphaX = fopen(fullfile(aModel.folder,'alphaX.dat'),'a');
		fAlphaY = fopen(fullfile(aModel.folder,'alphaY.dat'),'a');
		fAlphaZ = fopen(fullfile(aModel.folder,'alphaZ.dat'),'a');

		fBetaX = fopen(fullfile(aModel.folder,'betaX.dat'),'a');
		fBetaY = fopen(fullfile(aModel.folder,'betaY.dat'),'a');
		fBetaZ = fopen(fullfile(aModel.folder,'betaZ.dat'),'a');

		fResidual = fopen(fullfile(aModel.folder,'residual.dat'),'w');


	end

    
    
    constantX = reshape(xParameters(1,:),aStudy.dim(1),aStudy.dim(2),nSlicesChunk);
    constantY = reshape(yParameters(1,:),aStudy.dim(1),aStudy.dim(2),nSlicesChunk);
    constantZ = reshape(zParameters(1,:),aStudy.dim(1),aStudy.dim(2),nSlicesChunk);
           
    alphaX = reshape(xParameters(2,:),aStudy.dim(1),aStudy.dim(2),nSlicesChunk);
    alphaY = reshape(yParameters(2,:),aStudy.dim(1),aStudy.dim(2),nSlicesChunk);
    alphaZ = reshape(zParameters(2,:),aStudy.dim(1),aStudy.dim(2),nSlicesChunk);
            
    betaX = reshape(xParameters(3,:),aStudy.dim(1),aStudy.dim(2),nSlicesChunk);
    betaY = reshape(yParameters(3,:),aStudy.dim(1),aStudy.dim(2),nSlicesChunk);
    betaZ = reshape(zParameters(3,:),aStudy.dim(1),aStudy.dim(2),nSlicesChunk);

   
    % Save parameters
    fwrite(fConstantX,constantX,'single');
	fwrite(fConstantY,constantY,'single');
	fwrite(fConstantZ,constantZ,'single');

	fwrite(fAlphaX,alphaX,'single');
	fwrite(fAlphaY,alphaY,'single');
	fwrite(fAlphaZ,alphaZ,'single');

	fwrite(fBetaX,betaX,'single');
	fwrite(fBetaY,betaY,'single');
	fwrite(fBetaZ,betaZ,'single');

    % Last chunk? Close files
	if iChunk == nChunks
		fclose(fConstantX);
		fclose(fConstantY);
		fclose(fConstantZ);

		fclose(fAlphaX);
		fclose(fAlphaY);
		fclose(fAlphaZ);

		fclose(fBetaX);
		fclose(fBetaY);
		fclose(fBetaZ);
    end
    
    
    % Update the progress bar
    try
	waitbar(iChunk/nChunks, fitBar);
    end
    
    
% End loop over chunks
end

%toc
%% Save mean residual
fwrite(fResidual,meanResidual,'single');
fclose(fResidual);

try
	close(fitBar);
end

reset(gpuDevice);

%cudaReset;
aModel.study.patient.save;

end
