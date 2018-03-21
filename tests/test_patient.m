%% Main test function
function tests = test_patient
tests = functiontests(localfunctions);
end

%% Create test patient
function test_constructor(testCase);
aPatient = patient('test');
assert(isvarname('aPatient'));
assert(isa(aPatient,'patient'));
end


%% Try adding a study
function test_addStudy(testCase)

aPatient = testCase.TestData.aPatient;
dicomFolder = '/media/fiveDdata/dylan/5DCT_data/test/testData/';
bellowsDataFilename = '/media/fiveDdata/dylan/5DCT_data/test/testData/labview.txt';
aPatient.addStudy(dicomFolder,bellowsDataFilename,4);

% Test if study was created
assert(~isempty(aPatient.study));

% Test if study is a study
assert(strcmp(class(aPatient.study),'study')); 
end

function test_addStudy_nScans(testCase)

aPatient = testCase.TestData.aPatient;
dicomFolder = '/media/fiveDdata/dylan/5DCT_data/test/testData/';
bellowsDataFilename = '/media/fiveDdata/dylan/5DCT_data/test/testData/labview.txt';


bellowsDataFilenameNoExist = '/media/fiveDdata/dylan/5DCT_data/test/testData/labview_not_here.txt';
dicomFolderNoExist = '/media/fiveDdata/dylan/5DCT_data/test/testData_not_here';

testCase.verifyError(@() aPatient.addStudy(dicomFolder,bellowsDataFilename,-5),'MATLAB:expectedNonnegative');
testCase.verifyError(@() aPatient.addStudy(dicomFolder,bellowsDataFilename,0),'MATLAB:expectedNonZero');
testCase.verifyError(@() aPatient.addStudy(dicomFolder,bellowsDataFilename,nan),'MATLAB:expectedNonNaN');

testCase.verifyError(@() aPatient.addStudy(dicomFolder,bellowsDataFilenameNoExist,4),'FiveD:FileNotFound');

end

%% Create test patient
function setup(testCase)
aPatient = patient('test');
testCase.TestData.aPatient = aPatient;
end

%% Delete test patient
function teardown(testCase)
testCase.TestData = rmfield(testCase.TestData,'aPatient');
end
