function aScan = set_original(aScan, img, number, originalElementSpacing, dicoms, v, f, ekg, t, direction, studyUID)
% aScan.setOriginal(img, number, originalElementSpacing, dicoms, v, f, ekg, t, zPositions, direction, studyUID) sets a scan object as 'original', indicating that it was an acquired scan rather than one generated with the 5D model.  Associated with this scan is a 1x1x1 resolution image matrix img, a scan number number, the original resolution originalElementSpcing, a cell array of filenames of the DICOM files dicoms, a vector of breathing amplitudes v and rates f, and EKG values ekg, as well as times t, slice positions zPositions, a boolean indicating 1 if scan was caudocranial and 0 if craniocaudal direction, as well as the studyUID from the DICOM headers.

aScan.original = true;
aScan.dim = size(img);
aScan.originalDim = size(img);
aScan.img = img;
aScan.number = number;
aScan.filename = fullfile(aScan.parent.folder, sprintf('%02d.mat',aScan.number));

aScan.elementSpacing = originalElementSpacing;
aScan.originalElementSpacing = originalElementSpacing;
aScan.dicoms = dicoms;
aScan.v = v;
aScan.f = f;
aScan.ekg = ekg;
aScan.t = t;
aScan.direction = direction;

aScan.studyUID = studyUID;
end
