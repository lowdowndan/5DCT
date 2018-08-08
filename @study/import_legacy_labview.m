%IMPORT_LEGACY_LABVIEW Import surrogate, EKG, and x-ray on data saved 
% in original csv format.
%

function import_legacy_labview(aStudy, bellowsDataFilename)


aStudy.rawData = importdata(bellowsDataFilename);
aStudy.data = aStudy.rawData;

end

