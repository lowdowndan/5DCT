%% plot_histogram
%
% Generate histogram of breathing amplitudes

function plot_histogram(aBreath)

%% Get trace

vv = aBreath.study.data(aBreath.startInd:aBreath.stopInd, aBreath.study.channels.voltage);
%t = aBreath.study.data(aBreath.startInd:aBreath.stopInd, aBreath.study.channels.time);
t = 0.01:0.01:0.01*length(vv);
load('fivedcolor');
sampleRate = aBreath.study.sampleRate;

p5 = prctile(vv,5);
p85 = prctile(vv,85);
p95 = prctile(vv,95);


%% Plot trace annotated

traceFig = figure('visible','off');
plot(t,ones(size(t)) * p5, '--', 'linewidth', 3, 'color','k');
hold on
plot(t,ones(size(t)) * p85, '--', 'linewidth', 3, 'color',fivedcolor.orange);
plot(t,ones(size(t)) * p95, '--', 'linewidth', 3, 'color',fivedcolor.red);
plot(t,vv,'linewidth',1.525,'color',fivedcolor.blue);
xlim([t(1)-5 t(end) + 5]);
set(gca,'fontname','Droid Sans');
set(gca,'fontsize',32);
xlabel('Time (s)');
legend(' 5th', ' 85th', ' 95th', 'Location', 'best');
legend(gca,'boxoff')
ylabel('Breathing Amplitude (V)');
set(gcf,'units','normalized','Color',[1 1 1]);
set(gcf, 'Position', [0 0 0.99 0.99]); % Maximize figure.
f = getframe(gcf);
imwrite(f.cdata,fullfile(aBreath.folder,'documents','trace_annotated.png'),'png')
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
tCounts = counts .* sampleRate;
tTotal = length(vv) .* sampleRate;
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
	set(hBar(iBar),'facecolor',fivedcolor.gray);
end

for iBar = (lastLow + 1): (firstHigh - 1)
	set(hBar(iBar),'facecolor',fivedcolor.blue);
end


for iBar = firstHigh:(firstHigh95 - 1)
	set(hBar(iBar),'facecolor',fivedcolor.red);
end

for iBar = firstHigh95:length(vBinCenters)
	set(hBar(iBar),'facecolor',fivedcolor.gray);
end


% Legend
set(gca,'fontname','Droid Sans');
set(gca,'fontsize',32);

legend([hBar(1),hBar(lastLow + 1), hBar(firstHigh) ], ' Not Included',  ' < 85th', ' < 95th')
set(gcf, 'units', 'normalized'); 
set(gcf, 'Position', [0 0 0.99, 0.99]); 
legend(gca,'boxoff')
xlabel('Breathing Amplitude (V)');
ylabel('Percent Time');
set(barFig,'Color',[1 1 1]);
xlim([vBinCenters(1) - vWidth*2 vBinCenters(end) + vWidth *2])

f = getframe(gcf);
imwrite(f.cdata,fullfile(aBreath.folder,'documents','histogram.png'),'png')
