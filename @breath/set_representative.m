%% Generate a representative breath 
function aBreath = set_representative(aBreath, guiFlag)

load('fivedcolor');
chkmkdir(fullfile(aBreath.folder,'documents'));


if(~exist('guiFlag','var'))
    guiFlag = 0;
end

percentileInterval = aBreath.percentileInterval;
aStudy = aBreath.study;

%% Take entire breathing trace from start of 1st scan to end of last scan  
channels = aStudy.channels;
bellowsSampleRate = aStudy.data(2,channels.time) - aStudy.data(1,channels.time);

% Include n seconds before and after acquisition
relevanceWindow = 0;% 10 / .01;

breathTrace = aStudy.data(aStudy.startScan(1) - relevanceWindow :aStudy.stopScan(end) + relevanceWindow,channels.voltage);
t = aStudy.data(aStudy.startScan(1) - relevanceWindow :aStudy.stopScan(end) + relevanceWindow,channels.time);
t = t - t(1);


%% Select relevant region of breathing trace
if(guiFlag)

% Plot, get user to select subset of points to use
subsetFig = figure;
traceLine = plot(t,breathTrace,'color',fivedcolor.blue,'linewidth',1.5);
set(subsetFig.CurrentAxes, 'fontsize', 20);
set(subsetFig,'units','normalized','position', [0.1000    0.1000    0.8100    0.8100]);
title('Select a region of the breathing trace.');
xlabel('Time (s)', 'fontsize',20);
ylabel('Bellows Voltage (V)', 'fontsize', 20);
pointList = selectdata('Action','list','SelectionMode','Rect','Identify','on');
subsetFig.Color = [1 1 1];
indStart = min(pointList);
indEnd = max(pointList);

% Plot selected subset
hold on
traceLine.Color = fivedcolor.gray;
plot(t(indStart:indEnd),breathTrace(indStart:indEnd),'linewidth',1.5,'color',fivedcolor.blue);
legend('Breathing Trace','Selected subset', 'Location','NE');
f = getframe(gcf);
imwrite(f.cdata,fullfile(aBreath.folder,'documents','trace.png'),'png');
%print(fullfile(aStudy.folder,'documents','trace.png'),'-dpng');

%close(subsetFig);
% Hide figure for later saving
set(subsetFig, 'Visible', 'off');

% Trim breathing trace
t = t(indStart:indEnd);
breathTrace = breathTrace(indStart:indEnd);
t = t - t(1);  
end


%flowTrace = getFlow_sg(breathTrace);

%% Get percentile values 
pMin = prctile(breathTrace, percentileInterval(1));
pMax = prctile(breathTrace, percentileInterval(2));

%% Detect peaks and valleys
[~, extrema] = breath.detect_peaks_valleys(breathTrace, aStudy.sampleRate);

extremaFig = figure;
traceLine = plot(t,breathTrace,'color',fivedcolor.blue,'linewidth',1.5);
set(extremaFig.CurrentAxes, 'fontsize', 20);
set(extremaFig,'units','normalized','position', [0.1000    0.1000    0.8100    0.8100]);
xlabel('Time (s)', 'fontsize',20);
ylabel('Bellows Voltage (V)', 'fontsize', 20);
hold on;

% If gui flag was passed, plot and have user adjust/confirm
if(guiFlag)

	title('Verify extrema positions');
	hExtrema = cell(length(extrema),1);
	xx = xlim;
	yy = ylim;
	
	for iExtrema = 1:length(extrema)
	    
	hExtrema{iExtrema} = impoint(gca, t(extrema(iExtrema)), breathTrace(extrema(iExtrema)));
	setColor(hExtrema{iExtrema},'red');
	
	if(iExtrema == length(extrema))
	
	hExtreme = hExtrema{iExtrema};

	fConfirm = @(x,y) hExtreme.resume;
	fNew = @(x,y) setColor(impoint(gca, xx(1) + .05 * range(xx), yy(2) - .05 * range(yy)), 'r');
	
	buttonNew = uicontrol('Parent',extremaFig,'Style','pushbutton','String','Add point','FontWeight','bold','Units','normalized','Position',[0.0 .8  0.08 0.1],'Visible','on', 'callback', fNew);
	buttonConfirm = uicontrol('Parent',extremaFig,'Style','pushbutton','String','Confirm','FontWeight','bold','Units','normalized','Position',[0.0 .9  0.08 0.1],'Visible','on', 'callback', fConfirm);
	hExtreme.wait;  
	end
	% end loop over extrema
	end


	%  Hide buttons
	buttonConfirm.Visible = 'off';
	buttonNew.Visible = 'off';
%	extremaFig.Visible = 'off';

	% Update list of extrema
	extremaChildren = get(gca,'Children');
	removeMask = false(length(extremaChildren),1);

	for iExtrema = 1:length(extremaChildren)
		if ~strcmp('matlab.graphics.primitive.Group',class(extremaChildren(iExtrema)))
			removeMask(iExtrema) = 1;
		end
	end

	extremaChildren(removeMask) = [];
	extremaNew = zeros(length(extremaChildren),1);


	for iExtrema = 1:length(extremaChildren)
		extremaNew(iExtrema) = extremaChildren(iExtrema).Children(1).XData;
		extremaChildren(iExtrema).delete;
	end

	extremaNew = round(extremaNew / bellowsSampleRate);
	extremaNew = sort(extremaNew,'ascend');
	extremaNew = unique(extremaNew);
	extremaNew = extremaNew + 1;

	extrema = extremaNew;

end

% Plot extrema positions
for iExtrema = 1:length(extrema)
	plot(t(extrema(iExtrema)), breathTrace(extrema(iExtrema)), 'ro', 'linewidth',1.5);
end

title('Extrema positions');
% end guiFlag if

extremaFig.Color = [1 1 1];
extremaFig.Visible = 'off';


%% Segment breaths
breaths = cell(length(extrema) - 1, 1);

for iBreath = 1: length(breaths)
breaths{iBreath} = breathTrace(extrema(iBreath) : extrema(iBreath + 1));
end

%% Discard outliers in amplitude and period
amplitudes = cellfun(@range,breaths);
periods = cellfun(@length,breaths);


avgAmplitude = mean(amplitudes);
avgPeriod = mean(periods);

stdAmplitude = std(amplitudes);
stdPeriod = std(periods);

% Set tolerance to 2 standard deviations on either side of the mean
tol = 2;

discard = amplitudes > avgAmplitude + (stdAmplitude * tol);
discard = discard + (amplitudes < avgAmplitude - (stdAmplitude * tol));
discard = discard + (periods > avgPeriod + (stdPeriod * tol));
discard = discard + (periods < avgPeriod - (stdPeriod * tol));
discard = logical(discard);

breaths(discard) = [];

%% Normalize breaths

breaths = cellfun(@mat2gray,breaths,'uni',0);

%% Get amplitudes of representative breath

representativeBreath = zeros(round(avgPeriod),1);
pointAmplitudes = zeros(length(breaths),1);

for iPoint = 1: length(representativeBreath)
    
    pointPhase = iPoint / length(representativeBreath);
    
    for iBreath = 1:length(breaths)
        % Spline interpolation for smoothing
        pointAmplitudes(iBreath) = interp1([1:length(breaths{iBreath})] / length(breaths{iBreath}), breaths{iBreath}, pointPhase, 'spline');
    end

    representativeBreath(iPoint) = mean(pointAmplitudes);
    
end

%% Scale representative breath
representativeBreath = mat2gray(representativeBreath);

representativeBreath = representativeBreath * (abs(pMin - pMax));

% Shift representative breath so that min is at p5 (more negative = inhale)
rMin = min(representativeBreath(:));
shift = pMin - rMin;

representativeBreath = representativeBreath + shift; 

%% Smooth representative breath
representativeBreath = study.smooth(representativeBreath);

%% Plot representative breath
avgBreathFig = figure;
plot([.01:.01:.01 * length(representativeBreath)],representativeBreath, 'linewidth',1.5, 'color', fivedcolor.blue);
title('Representative Breath');
set(avgBreathFig.CurrentAxes, 'fontsize', 20);
set(avgBreathFig,'units','normalized','position', [0.1000    0.1000    0.8100    0.8100]);
xlabel('Time (s)', 'fontsize',20);
ylabel('Bellows Voltage (V)', 'fontsize', 20);

% Save
avgBreathFig.Color = [1 1 1];
f = getframe(avgBreathFig);
imwrite(f.cdata,fullfile(aBreath.folder,'documents','representativeBreath.png'),'png','WriteMode','overwrite');
savefig(avgBreathFig, fullfile(aBreath.folder,'documents','representativeBreath'));
close(avgBreathFig);


%% Plot representative breath in context
avgBreathContextFig = figure;
tStart = .1 * length(breathTrace) * .01;
plot([tStart + .01:.01:tStart + .01 * length(representativeBreath)],representativeBreath,'linewidth',1.5, 'color', fivedcolor.blue);
set(avgBreathContextFig.CurrentAxes, 'fontsize', 20);
set(avgBreathContextFig,'units','normalized','position', [0.1000    0.1000    0.8100    0.8100]);
title('Representative Breath');
xlabel('Time (s)', 'fontsize',20);
ylabel('Bellows Voltage (V)', 'fontsize', 20);
hold on;
plot(t - t(1),breathTrace,'--','color',fivedcolor.gray,'linewidth',1.5);

% Save
avgBreathContextFig.Color = [1 1 1];
f = getframe(avgBreathContextFig);
imwrite(f.cdata,fullfile(aBreath.folder,'documents','representativeBreathContext.png'),'png','WriteMode','overwrite');
savefig(avgBreathContextFig, fullfile(aBreath.folder,'documents','representativeBreathContext'));
close(avgBreathContextFig);

%% Save selection figure

if (guiFlag)

subsetFig.Color = [1 1 1];
f = getframe(subsetFig);
imwrite(f.cdata,fullfile(aBreath.folder,'documents','traceSubset.png'),'png','WriteMode','overwrite');
close(subsetFig)

end

%% Save extrema figure
f = getframe(extremaFig);
imwrite(f.cdata,fullfile(aBreath.folder,'documents','extrema.png'),'png','WriteMode','overwrite');

%% Modify breath object
aBreath.v = representativeBreath(:);

aBreath.t = aBreath.study.sampleRate: aBreath.study.sampleRate : aBreath.study.sampleRate * length(representativeBreath);
aBreath.t = aBreath.t(:);

aBreath.f = study.get_flow(aBreath.v,aBreath.study.sampleRate);
aBreath.f = aBreath.f(:);

aBreath.extrema = extrema;

if(guiFlag)
aBreath.startInd = indStart;
aBreath.stopInd = indEnd;
else
aBreath.startInd = aStudy.startScan(1) - relevanceWindow;
aBreath.stopInd = aStudy.stopScan(end) + relevanceWindow;
end


% Fix offset
aBreath.startInd = aBreath.startInd + aBreath.study.startScan(1) - 1;
aBreath.stopInd = aBreath.stopInd + aBreath.study.startScan(1) - 1;

%% Save
aBreath.study.patient.save;
