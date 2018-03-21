%% smooth_sg: Smooth using Savitsky-Golay filtering
% vSmooth = smooth_sg(v) returns a smoothed vector using the default
% moving window size of 50 and polynomial order of 3.
%
% vSmooth = smooth_sg(v,windowSize,n) returns a smoothed vector
% using moving window size windowSize and polynomial order n. Window
% size must be even.
function vSmooth = smooth_sg(varargin)

if nargin < 3
    n = 3;
end

if nargin < 2
    windowSize = 50;
else
    windowSize = varargin{2};
end


v = varargin{1};

if mod(windowSize,2)
    error('Window size must be even.');
end

fc = sgsdf([-windowSize/2 : windowSize/2],n,0);
vPad = zeros(length(v) + windowSize, 1);
vPad(1:(windowSize/2) - 1) = v(1);
vPad(end - (windowSize/2): end) = v(end);
vPad((windowSize/2):end - (windowSize/2) - 1) = v;

vSmooth = conv(vPad,flipdim(fc,2),'same');
vSmooth = vSmooth((windowSize/2) + 1:end - windowSize/2);


