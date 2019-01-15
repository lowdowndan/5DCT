%% QA Script


%% Generate qa patient object
aPatient = patient('qa');

aPatient.add_study;
aPatient.study.synchronize;
aPatient.study.import_scans;
aScan = aPatient.study.get_scan(1);
aScan.resample([1 1 1.5]);

scanFolder = fullfile(fileparts(aScan.filename),'qa_export');
aScan.write_dicom_anon(scanFolder,'QA','QA','QA1234567', dicomuid, sprintf('QA Image Export Test Scan %02d', aScan.number));
scan.push(scanFolder);

% Clear files
fileList = dir(scanFolder);
fileList = setdiff({fileList.name},{'.','..'});

if(~isempty(fileList))
	for iFile = 1:length(fileList)
		rmCmd = ['rm "' fullfile(scanFolder,fileList{iFile}) '"'];
		system(rmCmd);
	end
end






%aPatient.save;






