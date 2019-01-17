%% generate_scan

function aScan = generate_scan(aModel, v, f, studyUID, description, img)


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
