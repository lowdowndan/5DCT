
function setAcquisitionDate(aStudy,dicomTable)
%% Set acquisition date
dateHeader = dicominfo(dicomTable{1,1});
aStudy.date = datetime(dateHeader.AcquisitionDate,'InputFormat','yyyyMMdd');

dicomTime = dateHeader.AcquisitionTime;
hours = str2double(dicomTime(1:2));
minutes = str2double(dicomTime(3:4));
aStudy.date.Hour = hours;
aStudy.date.Minute = minutes;

