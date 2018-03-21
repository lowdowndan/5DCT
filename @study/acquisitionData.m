%% acquisitionData: Prepare data and plots for report generation

function reportData = acquisitionData(aStudy)

chkmkdir(fullfile(aStudy.folder,'report'));

%% Set amplitude range
reportData.pMin = 5;
reportData.pMax = 85;


%% Get dicom info
aScan = aStudy.getScan(1);
header = dicominfo(aScan.dicoms{1});

% Pitch
if isfield(header,'TableFeedPerRotation') && isfield(header,'TotalCollimationWidth');
reportData.pitch = header.TableFeedPerRotation / header.TotalCollimationWidth;
end

% Exposure
reportData.exposure = header.Exposure;

% Scan length
reportData.scanLength = header.SliceThickness * numel(aScan.dicoms) / 10;

% Estimated Dose
if isfield(header,'CTDIvol')

	reportData.ctdivol = header.CTDIvol * aStudy.nScans;

	kFactor = .014;
	reportData.eDose = reportData.ctdivol * reportData.scanLength *kFactor;
end

% Patient information

reportData.first = header.PatientName.GivenName;
reportData.last = header.PatientName.FamilyName;
reportData.mrn = header.PatientID;

%% Generate trace plot
start = aStudy.startScan(1);
stop = aStudy.stopScan(end);
vChannel = aStudy.channels.voltage;
v = aStudy.data(start:stop,vChannel);
t = [.01 : .01 : .01 * length(v)];
% Smooth
vv =  -1 * smooth_sg(v);

% Get percentiles

p5 = prctile(vv,5);
p85 = prctile(vv,85);
p95 = prctile(vv,95);

traceFig = figure('visible','off');
plot(t,ones(size(t)) * p5, '--', 'linewidth', 3, 'color','k');
hold on
plot(t,ones(size(t)) * p85, '--', 'linewidth', 3, 'color',[0.8750 0.25 0.1094]);
plot(t,ones(size(t)) * p95, '--', 'linewidth', 3, 'color',[0.6172 0.1680 0.1250]);
plot(t,vv,'linewidth',1.525,'color',[0 0.3320 0.6484]);
xlim([-5 t(end) + 5]);
set(gca,'fontname','Droid Sans');
set(gca,'fontsize',32);
xlabel('Time (s)');
legend(' 5th', ' 85th', ' 95th', 'Location', 'best');
legend(gca,'boxoff')
ylabel('Breathing Amplitude (V)');
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
export_fig(fullfile(aStudy.folder,'report','tracePlot'),'-transparent');
close(traceFig);

%% Generate percent time histogram
barFig = figure('visible','off');

% Set bin centers
nBinsV = 20;
vRange = range(vv);
vWidth = vRange / nBinsV;

vBinCenters = [min(vv(:)) + (vWidth /2) : vWidth : max(vv(:)) + (vWidth/2)];
vBinRightEdges = vBinCenters + vWidth/2;
vBinLeftEdges = vBinCenters - vWidth/2;

counts = histc(vv, vBinLeftEdges );
tCounts = counts .* aStudy.sampleRate;
tTotal = length(vv) .* aStudy.sampleRate;
percentTime = tCounts ./ tTotal;
percentTime = percentTime .* 100;

tooLow = vBinRightEdges <= p5;
%tooLow = vBinLeftEdges < p5;


tooHigh = vBinLeftEdges > p85;
%tooHigh = vBinRightEdges > p85;

lastLow = find(tooLow,1,'last');
if isempty(lastLow)
    lastLow = 0;
end

firstHigh = find(tooHigh,1,'first');
tooHigh95 = vBinLeftEdges > p95;
firstHigh95 = find(tooHigh95,1,'first');

% reportData.t1_4 = sum(percentTime(tooLow));
% reportData.t5_85 = sum(percentTime(lastLow + 1:firstHigh - 1));
% reportData.t86_94 = sum(percentTime(firstHigh: firstHigh95 -1));
% reportData.t5_95 = sum(percentTime(lastLow + 1: firstHigh95 - 1));
% reportData.t96_100 = sum(percentTime(firstHigh95:end));

totalLength = length(vv);
reportData.t1_4 = (nnz(vv(vv < p5)) ./ totalLength) .* 100;
reportData.t5_85 = (nnz(vv(vv >= p5 & vv <= p85)) ./ totalLength) * 100;
reportData.t86_94 = (nnz(vv(vv > p85 & vv < p95)) ./ totalLength) * 100;
reportData.t5_95 = (nnz(vv(vv >= p5 & vv <= p95)) ./ totalLength) * 100;
reportData.t96_100 = (nnz(vv(vv > p95 & vv <= max(vv(:)))) ./ totalLength) * 100;

sum(percentTime(lastLow + 1:firstHigh - 1));


hBar = bar(vBinCenters,diag(percentTime), 'stacked');

% Colors
for iBar = 1:lastLow
	set(hBar(iBar),'facecolor',[0.7969 0.7969 0.7969]);
end

for iBar = (lastLow + 1): (firstHigh - 1)
	set(hBar(iBar),'facecolor',[0 0.3320 0.6484]);
end


for iBar = firstHigh:(firstHigh95 - 1)
	set(hBar(iBar),'facecolor',[0.6172 0.1680 0.1250]);
end

for iBar = firstHigh95:length(vBinCenters)
	set(hBar(iBar),'facecolor',[0.7969 0.7969 0.7969]);
end


% Legend
set(gca,'fontname','Droid Sans');
set(gca,'fontsize',32);

legend([hBar(1),hBar(lastLow + 1), hBar(firstHigh) ], ' Not Included',  ' < 85th', ' < 95th')
set(gcf, 'Position', get(0,'Screensize')); 
legend(gca,'boxoff')
xlabel('Breathing Amplitude (V)');
ylabel('Percent Time');
xlim([vBinCenters(1) - vWidth*2 vBinCenters(end) + vWidth *2])
export_fig(fullfile(aStudy.folder,'report','histogramPlot'),'-transparent');

