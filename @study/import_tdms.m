%IMPORT_TDMS Import surrogate, EKG, and x-ray on data saved 
% in LabVIEW .tdms format.
%

function import_tdms(aStudy, bellowsDataFilename)

%% Convert and read in
[convertedData,~,~,~,~]=convertTDMS(0,bellowsDataFilename);

%% Parse

% Get date and time
header = {convertedData.Data.Root.Property.Name};
dateInd = find(strcmp('DateTime', header));
dateTag = convertedData.Data.Root.Property(dateInd).Value;
aStudy.labviewDate = datetime(dateTag);


% Get measured data
header = {convertedData.Data.MeasuredData.Name};

% bellows time
bellowsTimeInd = find(strcmp('Serial/Time',header));
bellowsTime = convertedData.Data.MeasuredData(bellowsTimeInd).Data;

% bellows pressure
bellowsInd = find(strcmp('Serial/Bellows',header));
bellows = convertedData.Data.MeasuredData(bellowsInd).Data;

% DAQ time
daqTimeInd = find(strcmp('DAQ/Time',header));
daqTime = convertedData.Data.MeasuredData(daqTimeInd).Data;

% X-ray on
xrayOnInd = find(strcmp('DAQ/X-ray On',header));
xrayOn = convertedData.Data.MeasuredData(xrayOnInd).Data;

% EKG
ekgInd = find(strcmp('DAQ/EKG',header));
ekg = convertedData.Data.MeasuredData(ekgInd).Data;


% Find common start time

startBellows = bellowsTime(1);
startDaq = daqTime(1);

% If DAQ starts first, choose the first daq sample after bellows signal
% starts
if(startBellows > startDaq)
startCommon = daqTime(find(daqTime > startBellows,1,'first'));

% If bellows starts first, choose the first DAQ sample
else
startCommon = startDaq;
end

% Find common end time

endBellows = bellowsTime(end);
endDaq = daqTime(end);


% If bellows ends first, choose the last daq sample before bellows signal
% ends
if(endBellows < endDaq)
endCommon = daqTime(find(daqTime < endBellows, 1, 'last'));


% If daq ends first, choose the last daq sample.
else
endCommon = endDaq;
end

% Common time
tCommon = startCommon : .01 : endCommon;

bellowsCommon = interp1(bellowsTime,bellows,tCommon,'spline');
xrayOnCommon = interp1(daqTime, xrayOn, tCommon, 'spline');
ekgCommon = interp1(daqTime, ekg, tCommon, 'spline');

% Normalize
tCommon = tCommon - tCommon(1) + .01;

% Save

data = [tCommon(:) bellowsCommon(:) xrayOnCommon(:) ekgCommon(:)];

aStudy.rawData = data;
aStudy.data = data;
aStudy.bellowsInhaleDirection = 1;

channels = struct('time',1,'voltage',2,'xrayOn',3,'ekg',4);
aStudy.channels = channels;







