%% plot_overlays
%
% Plot image overlays for DIR QA

function plot_overlays(aRegistration)

% Image overlays
ref = aRegistration.refScan;
nonRef = setdiff([1:aRegistration.study.nScans],ref);
refImg = aRegistration.study.get_image(ref);

corSlice = aRegistration.corSlice;
sagSliceL = aRegistration.sagSliceL;
sagSliceR = aRegistration.sagSliceR;



sRefCor = study.get_slice(refImg,1,corSlice);
sRefSagL = study.get_slice(refImg,2,sagSliceL);
sRefSagR = study.get_slice(refImg,2,sagSliceR);


% Loop over non-reference scans
lungWindow = [-1400 200];
overlayFig = figure('visible','off');

chkmkdir(fullfile(aRegistration.folder,'documents'));

for jScan = 1:length(nonRef)

	iScan = nonRef(jScan);

	% Load image
	img = load_nii(fullfile(aRegistration.folder,sprintf('%02d_deformed.nii',iScan)));
	img = img.img;

	% Get slices
	sCor = study.get_slice(img,1,corSlice);
	sSagL = study.get_slice(img,2,sagSliceL);
	sSagR = study.get_slice(img,2,sagSliceR);

	% Image overlay
	imgOverlay = imshowpair(mat2gray(sRefCor,lungWindow),mat2gray(sCor,lungWindow));
	axis image;
	set(gca,'xticklabel',[],'yticklabel',[]);
	imwrite(imgOverlay.CData,fullfile(aRegistration.folder,'documents',sprintf('%02d_cor.png',iScan)),'png');


	imgOverlay = imshowpair(mat2gray(sRefSagL,lungWindow),mat2gray(sSagL,lungWindow));
	axis image;
	set(gca,'xticklabel',[],'yticklabel',[]);
	imwrite(imgOverlay.CData,fullfile(aRegistration.folder,'documents',sprintf('%02d_sag_l.png',iScan)),'png');

	imgOverlay = imshowpair(mat2gray(sRefSagR,lungWindow),mat2gray(sSagR,lungWindow));
	axis image;
	set(gca,'xticklabel',[],'yticklabel',[]);
	imwrite(imgOverlay.CData,fullfile(aRegistration.folder,'documents',sprintf('%02d_sag_r.png',iScan)),'png');

end
close(overlayFig);
