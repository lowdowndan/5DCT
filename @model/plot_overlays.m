%% plot_overlays
%
% Plot image overlays for reconstruction QA

function plot_overlays(aModel)

% Image overlays
%ref = aModel.registration.refScan;
runScans = aModel.runScans;
%refImg = aModel.study.get_image(ref);

corSlice = aModel.registration.corSlice;
sagSliceL = aModel.registration.sagSliceL;
sagSliceR = aModel.registration.sagSliceR;

%% Verify that representative slices have been set
if(isempty(corSlice) || isempty(sagSliceL) || isempty(sagSliceR))
    error('Representative slices have not been set.  Run set_representative_slices method of registration class.');
end

%% Plots

% Loop over non-reference scans
lungWindow = [-1400 200];
overlayFig = figure('visible','off');

chkmkdir(fullfile(aModel.folder,'documents'));

for jScan = 1:length(runScans)

	iScan = runScans(jScan);

	% Load original image
    imgOg = aModel.study.get_image(iScan);
    
	% Get slices of original
	sRefCor = study.get_slice(imgOg,1,corSlice);
	sRefSagL = study.get_slice(imgOg,2,sagSliceL);
	sRefSagR = study.get_slice(imgOg,2,sagSliceR);
    
    % Load reconstruction
    imgSim = aModel.get_reconstruction(iScan);
    
    % Get slices of reconstruction
    sCor = study.get_slice(imgSim,1,corSlice);
	sSagL = study.get_slice(imgSim,2,sagSliceL);
	sSagR = study.get_slice(imgSim,2,sagSliceR);

	% Image overlay
	imgOverlay = imshowpair(mat2gray(sRefCor,lungWindow),mat2gray(sCor,lungWindow));
	axis image;
	set(gca,'xticklabel',[],'yticklabel',[]);
	imwrite(imgOverlay.CData,fullfile(aModel.folder,'documents',sprintf('recon_%02d_cor.png',iScan)),'png');


	imgOverlay = imshowpair(mat2gray(sRefSagL,lungWindow),mat2gray(sSagL,lungWindow));
	axis image;
	set(gca,'xticklabel',[],'yticklabel',[]);
	imwrite(imgOverlay.CData,fullfile(aModel.folder,'documents',sprintf('recon_%02d_sag_l.png',iScan)),'png');

	imgOverlay = imshowpair(mat2gray(sRefSagR,lungWindow),mat2gray(sSagR,lungWindow));
	axis image;
	set(gca,'xticklabel',[],'yticklabel',[]);
	imwrite(imgOverlay.CData,fullfile(aModel.folder,'documents',sprintf('recon_%02d_sag_r.png',iScan)),'png');

end
close(overlayFig);
