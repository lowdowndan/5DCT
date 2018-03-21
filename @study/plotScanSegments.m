function plotScanSegments(aStudy, noPlot)

%% Determine if plot should be displayed
if exist('noPlot','var')
    validateattributes(noPlot,{'logical','numeric'},{'numel',1,'finite','nonnegative'});
else
    noPlot = false;
end

if noPlot
        segmentPlot = figure('visible','off');

else
        segmentPlot = figure;

end
%% Plot

load fivedcolor

xrayOn = aStudy.rawData(:,aStudy.channels.xrayOn);
xrayColor = fivedcolor.black;

sampleNumbers = aStudy.dataRange(1) : aStudy.dataRange(2);
xrayOn = aStudy.rawData(sampleNumbers,aStudy.channels.xrayOn);
t = aStudy.rawData(sampleNumbers,1);
%t = t / 100;

startScanNorm = aStudy.startScan - aStudy.dataRange(1) + 1;
stopScanNorm = aStudy.stopScan - aStudy.dataRange(1) + 1;
hold on

plot(t(startScanNorm),xrayOn(startScanNorm),'gx','markersize',12,'linewidth',2);
plot(t(stopScanNorm),xrayOn(stopScanNorm),'rx','markersize',12,'linewidth',2);


plot(t,xrayOn,'color','k','linewidth',1.5);

xlim([t(startScanNorm(1)) - 5, t(stopScanNorm(end)) + 5]);
xlabel('Time (s)');
ylabel('X-Ray On Voltage');
set(gca,'fontsize',20);
legend('Start','Stop','location','NorthEastOutside')
set(segmentPlot,'units','normalized','position', [0.0000    0.0000    0.9900    0.9900],'color',[1 1 1]);


%% Save
f = getframe(gcf);
imwrite(f.cdata,fullfile(aStudy.folder,'documents','scanSegments.png'), 'png');

%print(segmentPlot,fullfile(aStudy.folder,'documents','scanSegments.png'), '-dpng');

