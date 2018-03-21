%% Average breath
function [time, voltage] = averageBreath(aStudy, runScans)


if nargin < 2
    runScans = [1:aStudy.nScans];
end

channels = aStudy.channels;
bellowsSampleRate = aStudy.data(2,channels.time) - aStudy.data(1,channels.time);

% Include n seconds before and after acquisition
relevanceWindow = 0;% 10 / .01;

breathTrace = aStudy.data(aStudy.startScan(runScans(1)) - relevanceWindow :aStudy.stopScan(runScans(end)) + relevanceWindow,channels.voltage);
t = aStudy.data(aStudy.startScan(runScans(1)) - relevanceWindow :aStudy.stopScan(runScans(end)) + relevanceWindow,channels.time);
flowTrace = getFlow_sg(breathTrace);

% Get percentile


if mean(breathTrace) < 0
p5 = prctile((breathTrace),95);
p85 = prctile((breathTrace),15);
p95 = prctile((breathTrace),5);
else
p5 = prctile(abs(breathTrace),5);
p85 = prctile(abs(breathTrace),85);
p95 = prctile(abs(breathTrace),95);
end

period = getPeriod(breathTrace);


% Phase estimate to segment breaths
tCrossings = [];
sPrev = 0;


periodHalf = floor(period/2);

% Begin ellipse tracking 1 period after beginning of trace
vStart = periodHalf + 1;

% Poincare sectioning
relativeAngle = pi/2;
nPoints = length(breathTrace) - vStart;
thetaV = zeros(nPoints,1);
phiV = zeros(nPoints,1);
aMat = zeros(nPoints,6);
xV = zeros(nPoints,1);
oldVals = [0, 0, 0];
baseline = zeros(nPoints,1);
tol = (1/8) * 2 * pi;


for iPoint = 1:length(breathTrace) - period - 1

    if iPoint == 1
        dPrev = 0;
        tCrossings = [];
    end
    
   pointIndex = iPoint + vStart;
  xWindow = [breathTrace(pointIndex - periodHalf: pointIndex + periodHalf), flowTrace(pointIndex - periodHalf: pointIndex + periodHalf)];
  [x, y, a, major, minor, theta] = ellipseFit_improved(xWindow);	
    
    if isempty(x)
        x = oldVals(1);
        y = oldVals(2);
        theta = oldVals(3);
    else
        
        if iPoint > 1 && abs(theta - thetaV(iPoint - 1)) > tol
	    theta = thetaV(iPoint - 1);
        end
        
    oldVals = [x, y, theta];
    end
    
    thetaV(iPoint) = theta;
    
    % Ellipse center as baseline
    baseline(iPoint) = x;
    phi = relativeAngle + theta;
    
    
    m = tan(phi);
    b = y - m*x;
    q1 = [x,y];
    q2 = [0, b];
    
    d = dPointLine(q1,q2,[breathTrace(pointIndex), flowTrace(pointIndex)]);
    
    if sign(d) - sign(dPrev) == 2  
        
        
            if isempty(tCrossings)
            tCrossings = pointIndex;
            else
                if tCrossings(end) < pointIndex - 100
                tCrossings = cat(1,tCrossings,pointIndex);
                end
            end
    end
    
    dPrev = d;
end
    
% Static Phase Estimate
vv = breathTrace(vStart + 1:end);
phaseV = zeros(nPoints,1);
for iPoint = 1:nPoints
    
    pointIndex = iPoint + vStart;
    tInd = find(tCrossings < pointIndex,1,'last'); 

    if isempty(tInd) ||  tInd == 1 
        phaseV(iPoint) = 0;
    elseif tInd == length(tCrossings)
	phaseV(iPoint) = 2*pi* (pointIndex - tCrossings(end))/(tCrossings(end) - tCrossings(end - 1));
    else
        phaseV(iPoint) = 2*pi* (pointIndex - tCrossings(tInd))/(tCrossings(tInd + 1) - tCrossings(tInd));
    end
end


% Subtract baseline drift
vv = vv - baseline;

% Segment breaths
breaths = cell(length(tCrossings) - 1,1);
for i = 2:length(tCrossings - 1)
    
    bv = vv(tCrossings(i - 1) - vStart + 1:tCrossings(i) -vStart); 
    bt = phaseV(tCrossings(i - 1) - vStart + 1:tCrossings(i) - vStart); 
    breaths{i - 1} = [bv(:), bt(:)];

end

% Discard outliers in amplitude and period
avgMax = mean(cellfun(@(x) max(x(:,1)), breaths));
stdMax = std(cellfun(@(x) max(x(:,1)), breaths));
avgMin = mean(cellfun(@(x) min(x(:,1)), breaths));
stdMin = std(cellfun(@(x) min(x(:,1)), breaths));
avgPeriod = mean(cellfun(@length, breaths));
stdPeriod = std(cellfun(@length, breaths));


amplitudeTolerance = 2;
periodTolerance = 2;
outlierInds = zeros(length(breaths),1);

for iBreath = 1:length(breaths)

	if max(breaths{iBreath}(:,1)) > avgMax + (amplitudeTolerance * stdMax)
		outlierInds(iBreath) = 1;

	elseif min(breaths{iBreath}(:,1)) < avgMin - (amplitudeTolerance * stdMin)
		outlierInds(iBreath) = 1;

	elseif length(breaths{iBreath}(:,1)) > avgPeriod + (periodTolerance * stdPeriod)
		outlierInds(iBreath) = 1;

	elseif length(breaths{iBreath}(:,1)) < avgPeriod - (periodTolerance * stdPeriod)
		outlierInds(iBreath) = 1;
		
	end

end
outlierInds = logical(outlierInds);
breaths(outlierInds) = [];


% Form template by taking average value at each phase
tTemplate = [bellowsSampleRate: bellowsSampleRate : period * 0.01 ];
pTemplate = [(2*pi) / length(tTemplate) : (2*pi) / length(tTemplate): 2 * pi]; 
vTemplate = zeros(size(pTemplate));


bvs = zeros(length(breaths) - 1,1);
bInds = 1:length(breaths) - 1;
for iPoint = 1:length(pTemplate)
    
    phase = pTemplate(iPoint);
    
    for i = 1:length(bvs)
    bvs(i) = interp1(breaths{i + 1}(:,2),breaths{i + 1}(:,1),phase, 'pchip');
    end
    
    vTemplate(iPoint) = mean(bvs);
end

vTemplate = [vTemplate vTemplate vTemplate];
[peakVal, peakInd] = max(vTemplate);



voltageRaw = vTemplate(peakInd : peakInd + period - 1);
voltageRaw = smooth_sg(voltageRaw);

%scale = abs(p85 - p5) / range(voltageRaw(:));
scale = 1;
disp('Scale = 1.0');
voltage = voltageRaw * scale;

% shift and scale
shift = p5 - max(voltage(:));
voltage = voltage + shift;


voltage = smooth_sg(voltage);
time = tTemplate;

voltage = voltage(:)';
time = time(:)';

%% Plot average breath
avgBreathFig = figure;
set(avgBreathFig.CurrentAxes, 'fontsize', 20);
set(avgBreathFig,'units','normalized','position', [0.1000    0.1000    0.8100    0.8100]);
plot(time,voltage);
title('Average Breath');
xlabel('Time (s)', 'fontsize',20);
ylabel('Bellows Voltage (V)', 'fontsize', 20);
print(fullfile(aStudy.folder,'documents','averageBreath.png'),'-dpng');
