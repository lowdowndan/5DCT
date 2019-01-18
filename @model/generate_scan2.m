%% generate_scan

function aScan = generate_scan2(aModel, v, f, varargin)


%% Parse inputs

aParser = inputParser;

vv = aModel.study.data(aModel.study.startScan(aModel.runScans(1)):aModel.study.stopScan(aModel.runScans(end)),aModel.study.channels.voltage);
tol = (range(vv(:)) * .1);

vMin = min(vv(:)) - tol;
vMax = max(vv(:)) + tol;

aParser.addRequired('v', @(x) validateattributes(x,{'numeric'},{'finite','nonnan','scalar','>=', vMin, '<=', vMax}));
aParser.addRequired('f', @(x) validateattributes(x,{'numeric'},{'finite','nonnan','scalar'}));

aParser.addOptional('studyUID', dicomuid, @(x)validateattributes(x, {'char'},{'nonempty'}));
aParser.addOptional('img',nan, @(x)validateattributes(x,{'numeric'},{'size',aModel.study.dim, 'finite','nonnan'}));
aParser.addOptional('description', '', @(x)validateattributes(x,{'char'},{'nonempty'}));


aParser.parse(v,f,varargin{:});

v = aParser.Results.v;
f = aParser.Results.f;
studyUID = aParser.Results.studyUID;
img = aParser.Results.img;
description = aParser.Results.description;


%% Was image passed?
if(numel(img) == 1 && isnan(img))
	img = aModel.registration.get_average_image;
end


aScan = scan(aModel, aModel.study.acquisitionInfo, aModel.study.imagePositionPatient, aModel.study.zPositions);
aScan.set_derived(img, v, f, studyUID, description);
