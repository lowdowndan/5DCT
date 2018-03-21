%% getFlow
% Calculates flow
function flow = getFlow(varargin)
voltage = varargin{1};

if nargin > 1
sampleRate = varargin{2};
else
sampleRate = .01;
end


flow = nan(size(voltage));

% Vector case
if isvector(voltage)
    
voltageSmooth = smooth(voltage,25);
flow(2:end) = diff(voltageSmooth) ./ sampleRate;
flow(1) = voltageSmooth(2) - voltageSmooth(1);
flow(1) = flow(1) ./ sampleRate;
flow = smooth(flow,5);


    
else

% Matrix case.  Treat each column

    for i = 1:size(voltage,2)
        voltageSmooth = smooth(voltage(:,i),25);
        flow(2:end,i) = diff(voltageSmooth) ./ sampleRate;
        flow(1,i) = voltageSmooth(2) - voltageSmooth(1);
        flow(1,i) = flow(1,i) ./ sampleRate;
    end


end