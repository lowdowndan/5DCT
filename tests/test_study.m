%% Main test function
function tests = test_study
tests = functiontests(localfunctions);
end

function test_setChannels_valid(testCase)

aPatient = testCase.TestData.aPatient;
channelsValid = testCase.TestData.channels;

channelsValid.time = 1;
channelsValid.voltage = 2;
channelsValid.xrayOn = 5;
channelsValid.ekg = 4;

assert(isempty(aPatient.study(1).channels),'Invalid test setup');

aPatient.study(1).setChannels(channelsValid,1);
assert(isequal(aPatient.study(1).channels,channelsValid));
end

function test_setChannels_alias(testCase);

aPatient = testCase.TestData.aPatient;
channelsAlias = testCase.TestData.channels;

channelsAlias.time = 1;
channelsAlias.voltage = 2;
channelsAlias.xrayOn = 2;
channelsAlias.ekg = 3;

testCase.verifyError(@() aPatient.study(1).setChannels(channelsAlias,1), 'FiveD:InvalidDataChannel');
end

function test_setSampleRate_valid(testCase)

%Set up
aPatient = testCase.TestData.aPatient;
channelsValid.time = 1;
channelsValid.voltage = 2;
channelsValid.xrayOn = 5;
channelsValid.ekg = 4;
aPatient.study(1).setChannels(channelsValid,1);

assert(isempty(aPatient.study.sampleRate));
aPatient.study.setSampleRate;
assert(isequal(aPatient.study.sampleRate, .01));
end


function test_driftCorrect_valid(testCase)

%% Set channels (necessary for drift correction)
aPatient = testCase.TestData.aPatient;
channelsValid = testCase.TestData.channels;
channelsValid.time = 1;
channelsValid.voltage = 2;
channelsValid.xrayOn = 5;
channelsValid.ekg = 4;
aPatient.study(1).setChannels(channelsValid,1);

%% Set sample rate
aPatient.study.setSampleRate;

%% Set data range
dataRange = [20602 26735];
aPatient.study(1).setDataRange(dataRange);

%% Set scan segments
aPatient.study(1).setScanSegments;

trueValue = -5.0755e-04;
tolerance = .001e-04;
aPatient.study(1).driftCorrect;
assert(abs(trueValue - aPatient.study(1).drift) < tolerance);
end

function test_smoothVoltage_valid(testCase)
%% Set up
aPatient = testCase.TestData.aPatient;
channelsValid = testCase.TestData.channels;
channelsValid.time = 1;
channelsValid.voltage = 2;
channelsValid.xrayOn = 5;
channelsValid.ekg = 4;
aPatient.study(1).setChannels(channelsValid,1);
% Set sample rate
aPatient.study.setSampleRate;
% Set data range
dataRange = [20602 26735];
aPatient.study(1).setDataRange(dataRange);
% Set scan segments
aPatient.study(1).setScanSegments;
% Drift correct
aPatient.study(1).driftCorrect;

%% Test
raw = aPatient.study.data(:,aPatient.study.channels.voltage);
aPatient.study.smoothVoltage;
smoothed = aPatient.study.data(:,aPatient.study.channels.voltage);

% Verify that voltage actually changed
assert(~isequal(raw,smoothed));

% Verify that smoothed voltage is within 5% of range of input
tol = .05;

vDiff = abs(raw - smoothed);
vDiff = vDiff(aPatient.study.startScan(1): aPatient.study.stopScan(end));
vDiff = max(vDiff);

assert( vDiff < range(raw(aPatient.study.startScan(1) : aPatient.study.stopScan(end)) .* tol));

end


function test_getDataSegments(testCase)
%% Set up
aPatient = testCase.TestData.aPatient;
channelsValid = testCase.TestData.channels;
channelsValid.time = 1;
channelsValid.voltage = 2;
channelsValid.xrayOn = 5;
channelsValid.ekg = 4;
aPatient.study(1).setChannels(channelsValid,1);
% Set sample rate
aPatient.study.setSampleRate;
% Set data range
dataRange = [20602 26735];
aPatient.study(1).setDataRange(dataRange);
% Set scan segments
aPatient.study(1).setScanSegments;
% Drift correct
aPatient.study(1).driftCorrect;
% Smooth 
aPatient.study.smoothVoltage;

%% Test 
[scanBellowsVoltage, scanBellowsFlow, scanBellowsTime, scanBellowsEkg] = aPatient.study.getDataSegments;

% Outputs should have the same dimensions
assert(size(scanBellowsVoltage,2) == size(scanBellowsFlow,2) && size(scanBellowsVoltage,2) == size(scanBellowsTime,2) && size(scanBellowsVoltage,2) == size(scanBellowsTime,2));

% Verify 2 NaN per output due to shorter scans 
assert(nnz(isnan(scanBellowsVoltage(:))) == 2);
assert(nnz(isnan(scanBellowsTime(:))) == 2);
assert(nnz(isnan(scanBellowsFlow(:))) == 2);
assert(nnz(isnan(scanBellowsEkg(:))) == 2);
end

function test_getDicomTable(testCase)
%% Set up
aPatient = testCase.TestData.aPatient;
channelsValid = testCase.TestData.channels;
channelsValid.time = 1;
channelsValid.voltage = 2;
channelsValid.xrayOn = 5;
channelsValid.ekg = 4;
aPatient.study(1).setChannels(channelsValid,1);
% Set sample rate
aPatient.study.setSampleRate;
% Set data range
dataRange = [20602 26735];
aPatient.study(1).setDataRange(dataRange);
% Set scan segments
aPatient.study(1).setScanSegments;
% Drift correct
aPatient.study(1).driftCorrect;
% Smooth 
aPatient.study.smoothVoltage;
[scanBellowsVoltage, scanBellowsFlow, scanBellowsTime, scanBellowsEkg] = aPatient.study.getDataSegments;


% Verify dcmtk is added to path
pathTest = system('dcmdump');
assert(pathTest == 0, 'issue calling dcmtk binary dcmdump');

% Test
dicomTable = getDicomTable(aPatient.study.dicomFolder);

% Correct size?
assert(isequal(size(dicomTable),[1368 9]));

% No empty entries?
assert(nnz(isempty(dicomTable(:))) == 0);

% Correct data types?
assert(all(all(cellfun(@ischar, dicomTable(:,1:3)))));
assert(all(all(cellfun(@isnumeric, dicomTable(:,4:8)))));
assert(all(cellfun(@ischar, dicomTable(:,9))));

end

function test_setDate(testCase)
%% Set up
aPatient = testCase.TestData.aPatient;
channelsValid = testCase.TestData.channels;
channelsValid.time = 1;
channelsValid.voltage = 2;
channelsValid.xrayOn = 5;
channelsValid.ekg = 4;
aPatient.study(1).setChannels(channelsValid,1);
% Set sample rate
aPatient.study.setSampleRate;
% Set data range
dataRange = [20602 26735];
aPatient.study(1).setDataRange(dataRange);
% Set scan segments
aPatient.study(1).setScanSegments;
% Drift correct
aPatient.study(1).driftCorrect;
% Smooth 
aPatient.study.smoothVoltage;
[scanBellowsVoltage, scanBellowsFlow, scanBellowsTime, scanBellowsEkg] = aPatient.study.getDataSegments;
dicomTable = getDicomTable(aPatient.study.dicomFolder);

aPatient.study.setAcquisitionDate(aPatient.study.dicomTable);

assert(isa(aPatient.study.date,'datetime'));
end

%% Create test patient and study
function setup(testCase)

aPatient = patient('test');
dicomFolder = '/media/fiveDdata/dylan/5DCT_data/test/testData/';
bellowsDataFilename = '/media/fiveDdata/dylan/5DCT_data/test/testData/labview.txt';
aPatient.addStudy(dicomFolder,bellowsDataFilename,4);
channels = struct('time',0,'voltage',0,'xrayOn',0,'ekg',0);

testCase.TestData.aPatient = aPatient;
testCase.TestData.channels = channels;

clear aPatient;

end

%% Delete test patient
function teardown(testCase)
testCase.TestData = [];
end
