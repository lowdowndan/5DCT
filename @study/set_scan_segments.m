function aStudy = set_scan_segments(aStudy)

%% Check if scan segments have already been identified
%assert(isempty(aStudy.stopScan) && isempty(aStudy.startScan), 'Scan segments already set.');

%% Get relevant region of X-Ray On Signal
xrayOn = aStudy.rawData(:,aStudy.channels.xrayOn);
dataRange = aStudy.dataRange;

xrayOn = xrayOn(dataRange(1):dataRange(2));

%% Get indices of large negative and positive jump discontinuities in x-ray on signal
sz = 350;
sigma = 117;
xrayOnFiltered = study.filter_xrayOn(xrayOn, sz, sigma);

dV = .1;
tMax = .6;
tMin = .4;

startIndices = peakfinder(xrayOnFiltered,dV,tMin,-1,0);
stopIndices = peakfinder(xrayOnFiltered,dV,tMax,1,0);

% Check number of large voltage jumps against number of scans
scanMatch = (length(startIndices) == aStudy.nScans && length(stopIndices) == aStudy.nScans);
if(~scanMatch)
    
    % Clear invalid data range
    aStudy.dataRange = [];
    
    % Error out
    error('The number of scans computed from x-ray on signal does not match number of scans entered for this acquisition.');
end


if startIndices(1) > stopIndices(1)
    swapTemp = startIndices;
    startIndices = stopIndices;
    stopIndices = swapTemp;
    clear swapTemp;
end

startIndices = startIndices + 1;

% Save scan start and stop indices
aStudy.startScan = startIndices + dataRange(1) - 1;
aStudy.stopScan = stopIndices + dataRange(1) - 1;





end
