classdef acquisition < handle

properties
number
dicomFolder
folder
rawData
data
drift
v
f
s
ekg
calibrated_v
calibrated_f
startScan
stopScan
date
nScans
channels
orientation
sampleRate
dim
scans
registrationDir
status
dvfDir
mask
dvfFolder
volumeFit
bellowsDataFilename
comment
end



methods

function anAcquisition = acquisition(aPatient,dicomFolder,bellowsDataFilename,nScans)

	% Check inputs

	assert(ischar(dicomFolder),'Invalid path to folder containing acquired images.');
	assert(ischar(bellowsDataFilename),'Invalid path to bellows data file.');
	assert(isnumeric(nScans),'Invalid number of scans');
	
	% Set properties necessary to sync
	anAcquisition.dicomFolder = dicomFolder;
	anAcquisition.rawData = importdata(bellowsDataFilename);
	anAcquisition.nScans = nScans;
	anAcquisition.number = length(aPatient.acquisition) + 1;
	anAcquisition.folder = fullfile(aPatient.folder,sprintf('acquisition_%02d',anAcquisition.number));
	mkdir(anAcquisition.folder);
    anAcquisition.bellowsDataFilename = bellowsDataFilename;

	status = struct;
	status.channels = false;
	status.driftCorrected = false;
	status.synchronized = false;
	status.aligned = false;
	status.resampled = false;
end

function anAcquisition = getChannels(anAcquisition)
	chanSelect = figure;
	title('Enter channel numbers.');
	set(chanSelect,'units','normalized','position', [0.1000    0.1000    0.8100    0.8100]);
	hold on
	set(gca,'fontsize',20);

	plot(anAcquisition.rawData(:,2), 'r', 'linewidth',1);
	plot(anAcquisition.rawData(:,3), 'g', 'linewidth',1);
	plot(anAcquisition.rawData(:,4), 'b', 'linewidth',1);
	plot(anAcquisition.rawData(:,5), 'k', 'linewidth',1);
	set(gca,'xtick',[]);
	ylabel('Volts (V)');
	
	legend('2','3','4','5');
	
	
	% Get channels 
	channels = struct;
	channels.time = 1;

	channels.voltage = input('Bellows channel: ');
	assert((isnumeric(channels.voltage) && 1 < channels.voltage && channels.voltage <= 5),'Invalid channel number.');
	channels.xrayOn = input('X-Ray On channel: ');
	assert((isnumeric(channels.xrayOn) && 1 < channels.xrayOn && channels.xrayOn <= 5 && (channels.xrayOn ~= channels.voltage)),'Invalid channel number.');
	channels.ekg = input('EKG channel: ');
	assert((isnumeric(channels.ekg) && 1 < channels.ekg && channels.ekg <= 5),'Invalid channel number.');

	
	% Set channel data
	anAcquisition.channels = channels;
	close(chanSelect);

	% Set bellows sampling rate
	anAcquisition.sampleRate = anAcquisition.rawData(2,channels.time) - anAcquisition.rawData(1,channels.time);
	
	% Update status
	anAcquisition.status.channels = true;

	% Call for save of patient object
	notify(anAcquisition,'statusChange');

end
	
	anAcquisition = synchronize(anAcquisition);
	
	
	anAcquisition = cropConvert(anAcquisition);



end
events
statusChange
end

end
