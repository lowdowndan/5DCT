
function aPatient = add_study(aPatient,varargin)
% aPatient.add_study: Create a new study object and append it to array of studies associated with this patient.
%
%aPatient.addStudy(dicomFolder,bellowsDataFilename, nScans) appends a study
%object, whose free-breathing CT scans are stored in dicomFolder, and breathing
%surrogate data in bellowsDataFilename, with nScans number of scans.  
%
%If no arguments are provided, the user will be prompted to select dicomFolder and
%bellowsDataFilename and enter nScans.

if ~(nargin == 1 || nargin == 4)
error('Usage: aPatient.addStudy(dicomFolder,bellowsDataFilename,nScans)');
end

if nargin == 1

validateattributes(aPatient,{'patient'},{});

% Check if user has a starting directory saved
if(ispref('fiveD','studyStartDir'))
    studyStartDir = getpref('fiveD','studyStartDir');
else
    studyStartDir = '';
end


dicomFolder = uigetdir(studyStartDir,'Select folder containing the dicom files for this study.');
[bellowsDataFilename,bellowsPath] = uigetfile('*.*','Select the file containing the bellows measurement.',dicomFolder);
bellowsDataFilename = fullfile(bellowsPath,bellowsDataFilename);


%nScans = input('Enter number of scans: ');
userResp = inputdlg({'Number of scans acquired'}, 'Scans', [1 30]);

% Was nScans  provided?
if(isempty(userResp) || strcmp(userResp{1},''))
    waitfor(errordlg('Enter number of scans.','Error', 'modal'));
else
    
% Is nScans number valid?
nScans = str2double(userResp{1});
end

validateattributes(nScans,{'numeric'},{'real', 'nonnan', 'finite', 'integer', 'nonzero', 'nonnegative', 'numel', 1});
else

inputData = inputParser;
inputData.FunctionName = 'addStudy';
inputData.addRequired('dicomFolder', @(x) validateattributes(x,{'char'},{'nonempty'}));
inputData.addRequired('bellowsDataFilename', @(x) validateattributes(x,{'char'},{'nonempty'}));
inputData.addRequired('nScans', @(x) validateattributes(x,{'numeric'},{'real', 'nonnan', 'finite', 'integer', 'nonzero', 'nonnegative', 'numel', 1}));

inputData.parse(varargin{:});

dicomFolder = inputData.Results.dicomFolder;
bellowsDataFilename = inputData.Results.bellowsDataFilename;
nScans = inputData.Results.nScans;

assert(logical(exist(dicomFolder,'dir')), 'Directory containing DICOM files not found');
assert(logical(exist(bellowsDataFilename,'file')), 'FiveD:FileNotFound','LabVIEW data file not found.');
end

% Append to existing studies (if any)
nStudy = numel(aPatient.study) + 1;

if(nStudy == 1)
aPatient.study = study(aPatient,bellowsDataFilename, dicomFolder,nScans);		
else
aPatient.study(nStudy) = study(aPatient,bellowsDataFilename, dicomFolder,nScans);		
end

aPatient.save
end



