
function aSequence = set_reconstruction_points(aSequence)

aBreath = aSequence.breath;

v = aBreath.v;
f = aBreath.f;
t = aBreath.t;

exhaleAmplitudes = sort(aSequence.exhaleAmplitudes,'descend');
inhaleAmplitudes = sort(aSequence.inhaleAmplitudes,'ascend');

% Find point of maximum inspiration (assumes bellows signal has been
% flipped so that inhalation is more positive.  flip is done by the study
% class.
[~,maxInd] = max(v);

%% Calculate exhale phases

% Voltage from maximum inspiration to end
exhaleV = min(v(maxInd:end)) + (exhaleAmplitudes/100) .* range(v(maxInd:end));
[~,exhaleInds] = min(abs(bsxfun(@minus, exhaleV, v(maxInd:end))));
exhaleInds = exhaleInds + maxInd - 1;

% Find flow
exhaleF  = f(exhaleInds)';

%% Inhale

inhaleV = min(v(1:maxInd)) + (inhaleAmplitudes/100) .* range(v(1:maxInd)); 
[~,inhaleInds] = min(abs(bsxfun(@minus, inhaleV, v(1:maxInd))));
inhaleF = f(inhaleInds)';

%% Reconstruction

lower = num2str(aBreath.percentileInterval(1));
upper = num2str(aBreath.percentileInterval(2));

prefix = sprintf('%sth to %sth ',lower,upper);
reconFields = fieldnames(aSequence.reconstructionPoints)';
reconFields(2,:) = cell(size(reconFields,1),1);
reconFields = reconFields(:);

reconInhale = struct(reconFields{:});
reconExhale = struct(reconFields{:});

% Inhale
nPoints = numel(inhaleV);
for iPoint = 1:nPoints
    
    reconInhale(iPoint).v = inhaleV(iPoint);
    reconInhale(iPoint).f = inhaleF(iPoint);
    reconInhale(iPoint).state = 'inhale';
    reconInhale(iPoint).amplitude = inhaleAmplitudes(iPoint);
    reconInhale(iPoint).description = [prefix sprintf('%d%% In', inhaleAmplitudes(iPoint)) ];
    reconInhale(iPoint).seriesUID = dicomuid;
end

    

% Exhale
nPoints = numel(exhaleV);
for iPoint = 1:nPoints
    
    reconExhale(iPoint).v = exhaleV(iPoint);
    reconExhale(iPoint).f = exhaleF(iPoint);
    reconExhale(iPoint).state = 'exhale';
    reconExhale(iPoint).amplitude = exhaleAmplitudes(iPoint);
    reconExhale(iPoint).description = [prefix sprintf('%d%% Ex', exhaleAmplitudes(iPoint)) ];
    reconExhale(iPoint).seriesUID = dicomuid;
end

aSequence.reconstructionPoints = cat(2,reconInhale,reconExhale);
aSequence.model.study.patient.save;

    
    
