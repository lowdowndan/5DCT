function aStudy = surrogateLinearity(aStudy)

%% Output folder
mkdir(fullfile(aStudy.folder,'documents','surrogateLinearity'));

%% Prompt user for slice
aScan = aStudy.getScan(1);
img = aScan.getImage;

imgSag = squeeze(max(img,[],2));

selFig = figure;
imagesc(imgSag);
colormap default;
title('Select an axial slice');
set(gca,'fontsize',24,'xtick',[],'ytick',[]);

[axSlice, ~ ] = ginput(1);
axSlice = round(axSlice);
close(selFig);

%% Choose profile
selFig = figure;
imgAx = squeeze(img(:,:,axSlice));
imagesc(imgAx);
title('Select profile location');
set(gca,'fontsize',24,'xtick',[],'ytick',[]);
hProfile = imline(gca,[size(imgAx,2) / 2, 10; size(imgAx,2) / 2, 30]);
hProfile.setColor('r');
profile = hProfile.wait;
close(selFig);

profileCol = round(mean(profile(:,1)));
profileRows = round(profile(:,2));
profileRows = sort(profileRows,'ascend');

%% Get abdomen heights
abdomenHeights = zeros(aStudy.nScans,1);

% Suppress warnings
warning('off','stats:nlinfit:ModelConstantWRTParam');
warning('off','MATLAB:rankDeficientMatrix');


heightBar = waitbar(0,'Finding abdominal heights...');

profileFig = figure;
tight_subplot(1,3,.01);

for iScan = 1:aStudy.nScans


	% Lod image
	if(iScan ~= 1)
		aScan = aStudy.getScan(iScan);
		img = aScan.getImage;
	end

	% Take profile along selected line
	imgAx = squeeze(img(:,:,axSlice));
	imgProfile = imgAx(profileRows(1):profileRows(2), profileCol);


	% Show HU profile
	subplot(1,3,1);
	plot([profileRows(1):profileRows(2)], imgProfile, '--.');
	set(gca,'fontsize',14);
	ylabel('HU')
	xlabel('Row');
%	title(sprintf('Scan %02d',iScan));
	ylim([-1024 400]);


	% Show image location
	subplot(1,3,2);
	imagesc(imgAx);
	set(gca,'fontsize',14);
	set(gca,'xticklabel',[],'yticklabel',[]);
	axis image;
	colormap gray;
	hold on
	plot([profileCol profileCol], [profileRows(1), profileRows(2)],'r','linewidth',1.5);
	hold off


	drawnow;
	pause(.01);
	
    	set(profileFig, 'units','normalized');
	set(profileFig, 'position', [0.0323    0.1783    0.8849    0.5108]);
	set(profileFig, 'outerposition', [0.0323    0.1783    0.8849    0.5108]);
	set(profileFig,'color',[1 1 1]);

	% Find the jump in HU values corresponding to border of body/air
	if(iScan == 1)
	subplot(1,3,3);
	param = sigm_fit([profileRows(1) : profileRows(2)], imgProfile,[],[],1);
	initialParam = param;
	
	else

	subplot(1,3,3);
	param = sigm_fit([profileRows(1) : profileRows(2)], imgProfile,[],initialParam,1);
	set(gca,'fontsize',14);
	ylabel('HU')
	xlabel('Row');
%	title(sprintf('Scan %02d',iScan));
	ylim([-1024 400]);
	end

	abdomenHeights(iScan) = size(img,1) - param(3);

	subplot(1,3,3)
	hold on
	plot([param(3) param(3)], get(gca,'ylim'), '--','color','k','linewidth',1.5);
	hold off

	f = getframe(gcf);
	%print(fullfile(aStudy.folder,'documents','surrogateLinearity',sprintf('%02d.png',iScan)),'-dpng');
	imwrite(f.cdata,fullfile(aStudy.folder,'documents','surrogateLinearity',sprintf('%02d.png',iScan)),'png');


	try
		waitbar(iScan/aStudy.nScans, heightBar);
	end

end

% Restore warnings
warning('on','stats:nlinfit:ModelConstantWRTParam');
warning('on','MATLAB:rankDeficientMatrix');


try
	close(heightBar);
	close(profileFig);
end


%% Check linearity


linearityFig = figure; 

v = aStudy.v(axSlice,:);
plot(abdomenHeights,v,'k+','markersize',8,'linewidth',1.5);
set(linearityFig,'Position',[0.5161         0    0.4833    0.9108]);
set(gca,'fontsize',24);
xlabel('Abdominal height (mm)');
ylabel('Bellows (V)');
hold on

surrogateLine = polyfit(abdomenHeights(:),v(:),1);
hLine = refline(surrogateLine);
hLine.color = 'r';


r = corr(abdomenHeights,v);
title(sprintf('R = %0.2f',r));
set(linearityFig,'color',[1 1 1]);

f = getframe(linearityFig);
imwrite(f.cdata,fullfile(aStudy.folder,'documents','surrogatePlot.png'),'png');




