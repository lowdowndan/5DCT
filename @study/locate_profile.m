function [profileRows, profileCol, profileZ] = locate_profile(aStudy, img, zPositions, outputFolder)

% Open selection figure
proFig = figure;
proFig.Units = 'normalized';
proFig.Position =  [0.0974 0.0767 0.8052 0.5708];
tight_subplot(1,3,.1);

%% Select axial slice from sagittal MIP view
imgSag = squeeze(max(img,[],2));

selSag = subplot(1,3,1);
selAx = subplot(1,3,2);
selProf = subplot(1,3,3);

axes(selSag);
imagesc(imgSag);
axis image;
colormap(sqrt(gray))
title('Select axial slice');
set(gca,'fontsize',24,'xtick',[],'ytick',[]);

[axSlice, ~ ] = ginput(1);
axSlice = round(axSlice);
hold on
yy = get(gca,'ylim');
plot([axSlice axSlice], [yy(1) yy(2)], 'r', 'linewidth', 2);
title('');
pause(.5);

%selFrame = getframe(selFigSag);
%imwrite(selFrame.cdata, fullfile(outputFolder,'profileSagittal.png'),'png');
%close(selFigSag);

%% Prompt user for profile and confirm

% Dialog box parameters
question = 'Use this profile location?';
dlgTitle = 'Profile selection';
respNo = 'No';
respYes = 'Yes';
respNew = 'Select a new axial slice';

userResponse = respNo;

while(~strcmp(userResponse, respYes))
switch userResponse

	case respNo
		
	cla(selProf);
	% Choose profile
	axes(selAx);
	imgAx = squeeze(img(:,:,axSlice));
	imagesc(imgAx);
	axis image;
	title('Select profile');
	set(gca,'fontsize',24,'xtick',[],'ytick',[]);
	colormap(sqrt(gray));
	hProfile = imline(gca,[size(imgAx,2) / 2, 10; size(imgAx,2) / 2, 30]);
	hProfile.setColor('r');
	profile = hProfile.wait;

	title('');

	% Confirm
	profileCol = round(mean(profile(:,1)));
	profileRows = round(profile(:,2));
	profileRows = sort(profileRows,'ascend');
	profileZ = zPositions(axSlice);
	
	axes(selProf);
	imgProfile = imgAx(profileRows(1):profileRows(2), profileCol);
	plot([profileRows(1):profileRows(2)], imgProfile, '--.');
	set(gca,'fontsize',14);
	ylabel('HU')
	xlabel('Row');
	
	userResponse = questdlg(question, dlgTitle, respYes, respNo, respNew, respNo);

	case respNew

	delete(hProfile);
	cla(selProf);
	cla(selAx);

	axes(selSag);
	imagesc(imgSag);
	axis image;
	colormap(sqrt(gray))
	title('Select axial slice');
	set(gca,'fontsize',24,'xtick',[],'ytick',[]);
	
	[axSlice, ~ ] = ginput(1);
	axSlice = round(axSlice);
	hold on
	yy = get(gca,'ylim');
	plot([axSlice axSlice], [yy(1) yy(2)], 'r', 'linewidth', 2);
	title('');
	pause(.5);
	userResponse = respNo;

	case respYes
	end;
end

set(proFig,'color','w');
proFrame = getframe(proFig);
imwrite(proFrame.cdata, fullfile(outputFolder,'profileLocation.png'),'png', 'WriteMode','overwrite');
close(proFig);
