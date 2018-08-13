function aStudy = correct_drift(aStudy)

abdomenHeights = aStudy.abdomenHeights;
calibrationVoltages = aStudy.calibrationVoltages;
calibrationTimes = aStudy.calibrationTimes;

options = optimset('MaxIter',10000);
costFun = @(x)correlate_surrogate(x,abdomenHeights,calibrationVoltages,calibrationTimes);
drift = fminsearch(costFun,0,options);


%assert(isequal(sign(drift), 1), 'FiveD:DriftWrongSign', 'Drift value has wrong sign; bellows signal should become increasingly positive.');
maxDrift = -5e-03;



drifted = aStudy.data(:,aStudy.channels.voltage);
drifted = drifted - aStudy.data(:,1) * drift;
voltagesDrifted = calibrationVoltages - (drift * calibrationTimes);
driftedCorrelation = corr(voltagesDrifted,abdomenHeights);


% Validate

DriftMagTooLarge = (abs(drift) > abs(maxDrift));
DriftCorrBad = (abs(driftedCorrelation) - abs(aStudy.initialCorrelation) < 0);

if(DriftMagTooLarge)
    warning('Magnitude of drift correction exceeds maximum allowable value.');
end

if(DriftCorrBad)
    warning('Computed drift correction would reduce correlation.')
end

% Use raw bellows if drift can't be fixed

if(DriftMagTooLarge || DriftCorrBad)
    warning('Drift correction not applied.  Raw bellows signal will be used for model fitting.');
    drifted = aStudy.data(:,aStudy.channels.voltage);
    drift = 0;
    driftedCorrelation = aStudy.initialCorrelation;
end


%assert(abs(drift) <= abs(maxDrift), 'FiveD:DriftMagTooLarge', 'Magnitude of drift correction exceeds maximum allowable value.');
%assert((abs(driftedCorrelation) - abs(aStudy.initialCorrelation) >= 0), 'Drift correction reduced correlation');


% Store
aStudy.drift = drift;
aStudy.data(:,aStudy.channels.voltage) = drifted;
aStudy.driftedCorrelation = driftedCorrelation;


% Call for save of patient object
%notify(aStudy.patient,'statusChange');

end 

function correlationCoefficient = correlate_surrogate(x,abdomenHeights,calibrationVoltages,calibrationTimes)

drifted = calibrationVoltages - (calibrationTimes * x);
correlationCoefficient = corr(drifted,abdomenHeights);
correlationCoefficient = -correlationCoefficient;
end

