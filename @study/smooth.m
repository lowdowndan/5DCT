%% get_flow
% Calculates flow using Savitsky-Golay filter
% https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter

function vSmooth = smooth(v)


%% Verify that input is a vector

if(~isvector(v))
    error('Input must be a vector.');
end

%% Pad signal, extrapolate with cubic

% Window size of 51 (25 on each side of central point)
% Arbitrary sample period -- does not affect result
samplePeriod = .01;
t = [0:samplePeriod:samplePeriod * (length(v) - 1)];
tPad = [-25 * samplePeriod : samplePeriod : samplePeriod * (length(v) -1 + 25)];
vPad = interp1(t,v,tPad,'pchip','extrap');


%% Precomputed convolution coefficients

% Smoothing, window size of 51
smoothCoef = [
   -0.0266
   -0.0211
   -0.0158
   -0.0107
   -0.0058
   -0.0012
    0.0033
    0.0075
    0.0114
    0.0152
    0.0187
    0.0219
    0.0250
    0.0278
    0.0304
    0.0328
    0.0350
    0.0369
    0.0386
    0.0401
    0.0413
    0.0423
    0.0431
    0.0437
    0.0440
    0.0441
    0.0440
    0.0437
    0.0431
    0.0423
    0.0413
    0.0401
    0.0386
    0.0369
    0.0350
    0.0328
    0.0304
    0.0278
    0.0250
    0.0219
    0.0187
    0.0152
    0.0114
    0.0075
    0.0033
   -0.0012
   -0.0058
   -0.0107
   -0.0158
   -0.0211
   -0.0266];

%% Convolve 
vSmooth = conv(vPad,(smoothCoef),'valid');

