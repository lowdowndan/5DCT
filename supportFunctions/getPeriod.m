%% getPeriod Determine period of breathing trace
%
%   period = getPeriod(v) finds the breathing period using
%   fft.

function period = getPeriod(breathTrace)

[maxValue,indexMax] = max(abs(fft(breathTrace - mean(breathTrace))));
sampleFrequency = 100;
frequency = indexMax * (sampleFrequency / length(breathTrace));
period = 1/frequency;
period = round(period * sampleFrequency);