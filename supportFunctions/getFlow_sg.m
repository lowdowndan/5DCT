%% getFlow_sg
% Calculates flow using Savitsky-Golay filter
function flow = getFlow_sg(varargin)

voltage = varargin{1};
flow = nan(size(voltage));

%% Set Parameters
% Moving window indices
window = [-25:25];
% Polynomial degree
n = 2;

%% Get filter coefficients
fc = sgsdf(window,n,1,0,0);

%% Get sample rate
if nargin > 1
sampleRate = varargin{2};
else
sampleRate = .01;
end

%% Get derivative
windowWidth = (length(window) - 1)/2;

if isvector(voltage) 
% Vector case

        vPad = zeros(length(window) + length(voltage), 1);
        vPad(1:windowWidth) = voltage(1);
        vPad(end - windowWidth:end) = voltage(end);
        vPad(windowWidth + 1:end-windowWidth - 1) = voltage;
        fPad = conv(vPad,flipdim(fc,2),'same');
        flow = fPad(windowWidth + 1:end - windowWidth - 1);
else
% Matrix case

        
 for i = 1:size(voltage,2)
        
        vPad = zeros(length(window) + length(voltage(:,i)), 1);
        vPad(1:windowWidth) = voltage(1,i);
        vPad(end - windowWidth:end) = voltage(end,i);
        vPad(windowWidth + 1:end-windowWidth - 1) = voltage(:,i);
        
        fPad = conv(vPad,flipdim(fc,2),'same');
        flow(:,i) = fPad(windowWidth + 1:end - windowWidth - 1);
    end
    

end

flow = flow ./ sampleRate;
