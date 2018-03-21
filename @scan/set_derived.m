function aScan = set_derived(aScan, img, v, f, studyUID, seriesDescription)

aScan.original = false;
aScan.dim = size(img);
aScan.img = img;
aScan.filename = fullfile(aScan.parent.folder, sprintf('%s.mat',aScan.seriesUID));
aScan.elementSpacing = [1 1 1];
aScan.v = v;
aScan.f = f;
aScan.studyUID = studyUID;

if(exist('seriesDescription','var'))
    aScan.seriesDescription = seriesDescription;
end

aScan.studyDescription = '5D Clinical Protocol';

end
