function drifted = driftCorrect(aStudy)

% Verify that drift correction has not already been performed
assert(isempty(aStudy.drift),sprintf('Drift correction already performed: Bellows drift value is %.05f V/.01s',aStudy.drift));

% Correct for linear drift in the bellows voltage signal.
v = aStudy.rawData(aStudy.startScan(1):aStudy.stopScan(end),aStudy.channels.voltage);
t = [aStudy.sampleRate:aStudy.sampleRate:aStudy.sampleRate * length(v)];
v = v(:);
t = t(:);

drift = polyfit(t,v,1);

drifted = aStudy.rawData(:,aStudy.channels.voltage);
drifted = drifted - aStudy.rawData(:,1) * drift(1);

drift = -drift(1);
assert(isequal(sign(drift), -1), 'FiveD:DriftWrongSign', 'Drift value has wrong sign; bellows signal should become increasingly positive.');
maxDrift = 1e-03;
assert(abs(drift) <= abs(maxDrift), 'FiveD:DriftMagTooLarge', 'Magnitude of drift correction exceeds maximum allowable value.');


aStudy.drift = drift;
aStudy.data = aStudy.rawData;
aStudy.data(:,aStudy.channels.voltage) = drifted;

% Call for save of patient object
%notify(aStudy.patient,'statusChange');

end 
