function aStudy = set_data_range(aStudy, dataRange)

if exist('dataRange', 'var')

    validateattributes(dataRange, {'numeric'},{'real','finite','positive','numel',2,'<=',length(aStudy.rawData)});
    aStudy.dataRange = dataRange;
        
else


    
load('fivedcolor');

% Threshold x-ray on signal
xrayOn = aStudy.rawData(:,aStudy.channels.xrayOn);
xrayColor = fivedcolor.black;

% Plot x-ray on vs. time and get user input to select data range
selectionPlot = figure;
set(selectionPlot,'units','normalized','position', [0.1000    0.1000    0.8100    0.8100]);
ylim([min(xrayOn) - 1, max(xrayOn) + 1]);
hold on
plot(aStudy.rawData(:,aStudy.channels.time),xrayOn,'color',xrayColor);
xlabel('Time');
ylabel('Scaled X-Ray On Signal');
title(sprintf('Select a range that contains %d full scans.',aStudy.nScans),'FontSize',20);
pointList = selectdata('sel','r','Verify','off','Pointer','crosshair');
hold off
close(selectionPlot);

% Use only data within selected range
dataRange(1) = min(pointList);
dataRange(2) = max(pointList);

aStudy.dataRange = dataRange;
end

end
