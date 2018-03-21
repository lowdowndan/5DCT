
function set_acquisition_info(aStudy,dicomTable)
%% Set acquisition date
header = dicominfo(dicomTable{1,1});
aStudy.date = datetime(header.AcquisitionDate,'InputFormat','yyyyMMdd');

dicomTime = header.AcquisitionTime;
hours = str2double(dicomTime(1:2));
minutes = str2double(dicomTime(3:4));
aStudy.date.Hour = hours;
aStudy.date.Minute = minutes;

%% Set other information

% Date
acquisitionInfo.AcquisitionDate = header.AcquisitionDate;
acquisitionInfo.AcquisitionTime = header.AcquisitionTime;
acquisitionInfo.SeriesTime = header.SeriesTime;
acquisitionInfo.SeriesDate = header.SeriesDate;
acquisitionInfo.StudyTime = header.StudyTime;
acquisitionInfo.StudyDate = header.StudyDate;

% Modality Tags

acquisitionInfo.Modality = 'CT';
acquisitionInfo.ConvolutionKernel = header.ConvolutionKernel;
acquisitionInfo.DataCollectionDiameter = header.DataCollectionDiameter;
acquisitionInfo.DistanceSourceToDetector = header.DistanceSourceToDetector;
acquisitionInfo.DistanceSourceToPatient = header.DistanceSourceToPatient;
acquisitionInfo.FocalSpot = header.FocalSpot;
acquisitionInfo.GantryDetectorTilt = header.GantryDetectorTilt;

acquisitionInfo.ContentDate = header.ContentDate;
acquisitionInfo.ContentTime = header.ContentTime;

% DICOM UID
acquisitionInfo.FrameOfReferenceUID = header.FrameOfReferenceUID;
acquisitionInfo.ImageType = 'DERIVED\SECONDARY';
%a

% Patient
acquisitionInfo.ImageOrientationPatient = header.ImageOrientationPatient;
%acquisitionInfo.ImagePositionPatient = header.ImagePositionPatient;
%acquisitionInfo.ImagePositionPatient(3) = nan;
acquisitionInfo.TableHeight = header.TableHeight;
acquisitionInfo.ReferringPhysicianName = header.ReferringPhysicianName;

% Dose
acquisitionInfo.KVP = header.KVP;
acquisitionInfo.Exposure = header.Exposure;
acquisitionInfo.ExposureTime = header.ExposureTime;

% Patient Info
if(~strcmp(header.PatientID, aStudy.patient.id))
    warning('Patient ID entered does not match the ID on the DICOM headers.  Using entered ID.');
end
acquisitionInfo.PatientID = num2str(aStudy.patient.id);




acquisitionInfo.PatientName = header.PatientName;
acquisitionInfo.PatientPosition = header.PatientPosition;
acquisitionInfo.PatientBirthDate = header.PatientBirthDate;
acquisitionInfo.PatientSex = header.PatientSex;

% Scanner
acquisitionInfo.Manufacturer = header.Manufacturer;
acquisitionInfo.ManufacturerModelName = header.ManufacturerModelName;
acquisitionInfo.DeviceSerialNumber = header.DeviceSerialNumber;
acquisitionInfo.DateOfLastCalibration = header.DateOfLastCalibration;
acquisitionInfo.InstitutionName = 'UCLA';

% Rescale
acquisitionInfo.RescaleSlope = header.RescaleSlope;
acquisitionInfo.RescaleIntercept = header.RescaleIntercept;

%% Save
aStudy.acquisitionInfo = acquisitionInfo;


%% AS64 --> Positive Z is into bore
