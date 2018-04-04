function setAcquisitionDate(aStudy,dicomTable)
%% Set patient info
header = dicominfo(dicomTable{1,1});
aStudy.date = datetime(header.AcquisitionDate,'InputFormat','yyyyMMdd');

aStudy.patient.set('first', header.PatientName.GivenName);
aStudy.patient.set('last', header.PatientName.FamilyName);
aStudy.patient.set('mrn', header.PatientID);

