function slice(aRegistration)



aStudy = aRegistration.study;
refScan = aRegistration.refScan;
sliceFolder = aRegistration.sliceFolder;


%% Get dim

dim = aStudy.dim;
nSlices = dim(3);

%% Process DVFs

sliceBar = waitbar(0, 'Slicing deformation fields...');

for iDvf = aStudy.nScans

	% Read this DVF into memory
	if (iDvf ~= refScan)
	dvf = aRegistration.import_registration(fullfile(aRegistration.folder,sprintf('%02d',iDvf)));
	else
	dvf = zeros([aStudy.dim 3],'single');
	end


	% Save each axial slice
	for iSlice = 1:nSlices

		% If first DVF, create file
		if (iDvf == 1)
			fSlice = fopen(fullfile(sliceFolder,sprintf('%04d.dat',iSlice)),'w');
		% Otherwise, append this slice to file
		else
			fSlice = fopen(fullfile(sliceFolder,sprintf('%04d.dat',iSlice)),'a');
		end


		% Reshape
		% Write dvf data
		fwrite(fSlice,squeeze(dvf(:,:,iSlice,:)),'single');

		% Close file
		fclose(fSlice);
    end
    
	try
	waitbar(iDvf / aStudy.nScans,sliceBar);
    end
    
end


try
	close(sliceBar)
end

