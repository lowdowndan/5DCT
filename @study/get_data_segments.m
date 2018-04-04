function [scanBellowsVoltage, scanBellowsFlow, scanBellowsTime, scanEkg] = get_data_segments(aStudy)

dataRange = aStudy.dataRange;
stopIndices = aStudy.stopScan - dataRange(1) + 1;
startIndices = aStudy.startScan - dataRange(1) + 1;

bellowsTime = aStudy.data(dataRange(1):dataRange(2),aStudy.channels.time);
bellowsVoltage = aStudy.data(dataRange(1):dataRange(2),aStudy.channels.voltage);
bellowsFlow = study.get_flow(bellowsVoltage, aStudy.sampleRate);
ekg = aStudy.data(dataRange(1):dataRange(2),aStudy.channels.ekg);
maxScanLength = max(stopIndices - startIndices);

scanBellowsVoltage = arrayfun(@(x,y) bellowsVoltage(x:y - 1), startIndices, stopIndices, 'UniformOutput', false);
scanBellowsVoltage = cell2mat(cellfun(@(x) cat(1,x,nan(maxScanLength - length(x),1)), scanBellowsVoltage, 'UniformOutput',false));
scanBellowsVoltage = reshape(scanBellowsVoltage, maxScanLength, aStudy.nScans);

scanBellowsFlow = arrayfun(@(x,y) bellowsFlow(x:y - 1), startIndices, stopIndices, 'UniformOutput', false);
scanBellowsFlow = cell2mat(cellfun(@(x) cat(1,x,nan(maxScanLength - length(x),1)), scanBellowsFlow, 'UniformOutput',false));
scanBellowsFlow = reshape(scanBellowsFlow, maxScanLength, aStudy.nScans);

scanBellowsTime = arrayfun(@(x,y) bellowsTime(x:y-1), startIndices, stopIndices, 'UniformOutput', false);
scanBellowsTime = cell2mat(cellfun(@(x) cat(1,x,nan(maxScanLength - length(x),1)), scanBellowsTime, 'UniformOutput',false));
scanBellowsTime = reshape(scanBellowsTime, maxScanLength, aStudy.nScans);

scanEkg = arrayfun(@(x,y) ekg(x:y - 1), startIndices, stopIndices, 'UniformOutput', false);
scanEkg = cell2mat(cellfun(@(x) cat(1,x,nan(maxScanLength - length(x),1)), scanEkg, 'UniformOutput',false));
scanEkg = reshape(scanEkg, maxScanLength, aStudy.nScans);
end
