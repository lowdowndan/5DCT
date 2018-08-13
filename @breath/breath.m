classdef breath < handle

properties
end

properties(SetAccess = protected)
v
f
t
startInd
stopInd
extrema
uuid
study
folder
percentileInterval
pMin
pMax


end

properties(Access = protected)

end

methods
	
% Constructor
function aBreath = breath(aStudy)

aBreath.study = aStudy;
aBreath.uuid = char(java.util.UUID.randomUUID);
chkmkdir(fullfile(aStudy.folder,'breaths'));
aBreath.folder = fullfile(aStudy.folder,'breaths',aBreath.uuid);
chkmkdir(aBreath.folder);
aBreath.percentileInterval = [5 85];
end


% Set
function aBreath = set(aBreath,property,value)
aBreath.(property) = value;
end

% Get
function value = get(aBreath,property)
value = aBreath.(property);
end

% Save
function save(aBreath)
save(fullfile(aBreath.folder,'breath.mat'));
end


end

methods (Static)
[peaks, valleys] = detect_peaks_valleys(trace, sampleRate)

end

events
end

end



