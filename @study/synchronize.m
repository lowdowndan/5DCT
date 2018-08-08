function aStudy = synchronize(aStudy, refScan)

%% Verify that study has not already been synchronized;
if(strcmp(aStudy.status,'synchronized'))
error('This study has already been synchronized.  Overwriting synchronized data is not allowed.  Please create a new study.');
end


%% Default reference scan is 1
% TODO:
% Add reference scan selection
if(~exist('refScan','var'))
refScan = 1;
end
aStudy.refScan = refScan;

  
%% Select data channels
if(isempty(aStudy.channels))
	aStudy.set_channels;
end

%% Get inhalation direction, flip if necessary
if(isempty(aStudy.bellowsInhaleDirection))
    aStudy.set_bellows_inhale_direction;
end



%% Smooth voltage signal
if(isempty(aStudy.bellowsSmoothingWindow))
aStudy.data(:,aStudy.channels.voltage) = study.smooth(aStudy.data(:,aStudy.channels.voltage));
aStudy.bellowsSmoothingWindow = 51;
end

%% Set sample rate
if(isempty(aStudy.sampleRate))
    aStudy.set_sample_rate;
end

%% Prompt user for relevant region of breathing trace
if(isempty(aStudy.dataRange))
    aStudy.set_data_range;
  %  aStudy.patient.save;
end

%% Get scan start/end times
if(isempty(aStudy.startScan) || isempty(aStudy.stopScan))
 
     aStudy.set_scan_segments;
     % Plot and record scan segments
     aStudy.plot_scan_segments(1);
     aStudy.patient.save;

end

%% Get table of dicom tags:
% Table is structured as follows:
% | Filename | SeriesUID | Acquisition Time | Image Position Patient [X,Y,Z] | Slice Thickness | [X Spacing; Y Spacing] | [Rows; Columns] | ...
% [rescale slope; rescale intercept] |SOP Instance UID

% Save dicom table?
if exist(fullfile(aStudy.folder,'dicomTable.mat'))
load(fullfile(aStudy.folder,'dicomTable.mat'));
else
dicomTable = get_dicomTable(aStudy);
save(fullfile(aStudy.folder,'dicomTable.mat'),'dicomTable');
end

%% Parse dicom table and verify all scans have the same Z positions, pixel spacing, etc 
%[dicomTable, scanIDs, sliceCounts] = aStudy.processDicomTable(dicomTable);

if(isempty(aStudy.scanIDs) || isempty(aStudy.sliceCounts))
dicomTable = aStudy.verify_image_positions(dicomTable);

% Save updated dicom table
save(fullfile(aStudy.folder,'dicomTable.mat'),'dicomTable');

% Save patient
aStudy.patient.save;
end

%% Calibrate surrogate
if(isempty(aStudy.abdomenHeights) || isempty(aStudy.calibrationVoltages) || isempty(aStudy.calibrationTimes) || isempty(aStudy.drift))

% Initial estimate of 0 drift
calibrated = false;

while(calibrated == false)
  
% Calibrate    
%% TODO:
% Add interpolation if slice doesnt exist in all scans
aStudy.calibrate_surrogate(dicomTable);

% Report initial correlation
linearityFig = figure;
plot(aStudy.abdomenHeights,aStudy.calibrationVoltages,'k+','markersize',8,'linewidth',1.5);
set(linearityFig,'units','normalized');
set(linearityFig,'Position',[0         0    0.99    0.99]);
set(gca,'fontsize',24);
xlabel('Abdominal height (mm)');
ylabel('Bellows (V)');
hold on
surrogateLine = polyfit(aStudy.abdomenHeights,aStudy.calibrationVoltages,1);
hLine = refline(surrogateLine);
hLine.Color = 'r';
r = aStudy.initialCorrelation;
title(sprintf('R = %0.2f',r));
set(linearityFig,'color',[1 1 1]);
% Get confirmation from user
% Dialog box parameters
question = 'Confirm?';
dlgTitle = 'Surrogate calibration';
respNo = 'No';
respYes = 'Yes';

userResponse = questdlg(question,dlgTitle,respYes,respNo,respNo);

if(strcmp(userResponse,respYes))
calibrated = true;
% Drift correct
aStudy.correct_drift;

% Save linearity figure
f = getframe(linearityFig);
imwrite(f.cdata,fullfile(aStudy.folder,'documents','surrogatePlot.png'),'png');
close(linearityFig);


else
close(linearityFig);
% end userResposne strcmp
end
% end while loop
end
% end if(isempty(aStudy.abdomenHeights ...) loop
end


% Mark study as synchronized
aStudy.status = 'synchronized';

% Call for save of patient object
aStudy.patient.save;
end


