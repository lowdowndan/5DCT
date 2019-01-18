%% generate_scan

%function aScan = generate_scan2(aModel, v, f, studyUID, description, img)
function aScan = generate_scan2(aModel, v, f, varargin)



aParser = inputParser;

vv = aModel.study.data(aModel.study.startScan(aModel.runScans(1)):aModel.study.stopScan(aModel.runScans(end)),aModel.study.channels.voltage);
tol = (range(vv(:)) * .1);

vMin = min(vv(:)) - tol;
vMax = max(vv(:)) + tol;



aParser.addRequired('v', @(x)validateattributes(x,{'numeric'},{'finite','nonnan','scalar','>=', vMin, '<=', vMax));
aParser.addRequired('f', @(x)validateattributes(x,{'numeric'},{'finite','nonnan','scalar'}))

	aParser.


%isvalid_f = @(x) x < (max(aModel.study.f(:)) + (range(aModel.study.f(:)) * .1)) && x > (min(aModel.study.f(:)) - (range(aModel.study.f(:)) * .1)) || x == 0;

studyUID, description, img)



% Generate image if needed 
if(~exist('img','var'))
img = aModel.generate_image(v,f);
end

aScan = scan(aModel, aModel.study.acquisitionInfo, aModel.study.imagePositionPatient, aModel.study.zPositions);


if(~exist('description','var'))
	description = '5D Dervied Scan';
end

if(~exist('studyUID','var'))
	studyUID = dicomuid;
end

aScan.set_derived(img, v, f, studyUID, description);
