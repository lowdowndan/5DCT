%% get_flow
% Calculates flow using Savitsky-Golay filter
% https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter

function flow = get_flow(v, samplePeriod)


%% Verify that input is a vector

if(~isvector(v))
    error('Input must be a vector.');
end


%% Check if sample rate is specified

if(~exist('samplePeriod','var'))
    samplePeriod = .01;
    warning('No sample period specified.  Setting sample period to .01 s.')
end

%% Pad signal, extrapolate with cubic

% Window size of 9 (4 on each side of central point)

t = [0:samplePeriod:samplePeriod * (length(v) - 1)];
tPad = [-4 * samplePeriod : samplePeriod : samplePeriod * (length(v) -1 + 4)];
vPad = interp1(t,v,tPad,'pchip','extrap');


%% Precomputed convolution coefficients (window size of 9)
derivCoeff = [-86 142 193 126 0 -126 -193 -142 86] ./ 1188;

%% Convolve and scale

flow = conv(vPad,(derivCoeff),'valid');
flow = flow ./ samplePeriod;

%% Return row or column vector depending on input

if(iscolumn(v))
    flow = flow(:);
else
    flow = flow(:)';
end
