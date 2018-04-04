%% Detect peaks of free-breathing respiratory waveforms
%
% Based on:
% Lu, W., Nystrom, M.M., Parikh, P.J., Fooshee, D.R., Hubenschmidt, J.P.,
% Bradley, J.D. and Low, D.A., 2006. A semi‐automatic method for peak and valley
% detection in free‐breathing respiratory waveforms. Medical Physics, 33(10),
% pp.3634-3636.

function [peaks, valleys] = detect_peaks_valleys(trace, sampleRate)


if (~exist('sampleRate','var'))
	sampleRate = .01;
end

%% Estimate period
nSeconds = 15;
detrended = trace - mean(trace);
detrended = detrended(1: nSeconds / sampleRate);	

[maxValue,indexMax] = max(abs(fft(detrended)));
sampleFrequency = 1/sampleRate;
frequency = indexMax * (sampleFrequency / length(detrended));

period = 1/frequency;
period = round(period * sampleFrequency);


%% Calculate moving average curve
width = 2 * period;
kernel = ones(width,1);
kernel = kernel ./ width;

mac = conv(trace,kernel,'same');

mac(1:width) = mean(trace(1:width));
mac(end - width: end) = mean(trace(end-width:end));

%% Find intercepts of the MAC with the respiratory waveform

zci = zeros(0,2);
for iPoint = 1:length(trace)

	if(iPoint == 1 || iPoint == length(trace))

	else

		if trace(iPoint - 1) <= mac(iPoint - 1) && trace(iPoint) >= mac(iPoint)
			zci = cat(1,zci,[iPoint 1]);

		elseif trace(iPoint - 1) >= mac(iPoint - 1) && trace(iPoint) <= mac(iPoint)
			zci = cat(1,zci,[iPoint -1]);
		end

	end
end


%% Discard intercepts that are too short
minDist = period / 20;

discard = zeros(length(zci),1);
discard(2:end) =diff(zci(:,1));
discard = discard < (minDist);
discard = logical(discard);
zci(discard,:) = [];

%% Check labels of intercepts, make sure they alternate
discard = zeros(length(zci), 1);
for iIntercept = 1:length(zci)

	if(iIntercept == 1 || iIntercept == length(zci))

	else

		%% Are there two consecutive intercepts with the same label? --> discard 1st
		% For more than two consecutive intercepts with the same, label, keep only last

		if zci(iIntercept,2) == zci(iIntercept + 1, 2) 

			i = 1;
			while(zci(iIntercept,2 ) == zci(iIntercept + i, 2))
				discard(iIntercept + i - 1) = 1;
				i = i + 1;
			end
		end


	end

end

discard = logical(discard);
zci(discard,:) = [];

%% Verify that intercept labels alternate
for iIntercept = 1:length(zci) - 1
assert(zci(iIntercept,2) ~= zci(iIntercept + 1,2), sprintf('Consecutive intercepts with same label detected at location %d', iIntercept));
end

%% Get peaks
% Maximum between an up intercept and the following down intercept

peaks = [];
start = find(zci(:,2) == 1, 1, 'first');
stop = find(zci(:,2) == 1, 1, 'last');

% Is last point an up intercept?  If so only process up to second last intercept
if stop == length(zci)
	stop = stop - 2;

end

for iIntercept = start: 2: stop

	[~, ind ] = max(trace(zci(iIntercept,1): zci(iIntercept + 1,1)));
	ind = ind + zci(iIntercept,1) - 1;
	peaks = cat(1,peaks,ind);
end


%% Get valleys
% Minimum between a down intercept and the following up intercept

valleys = [];

start = find(zci(:,2) == -1, 1, 'first');
stop = find(zci(:,2) == -1, 1, 'last');

% Is last point a down intercept?  If so only process up to second last intercept
if stop == length(zci)
	stop = stop - 2;
end

for iIntercept = start: 2: stop

	[~, ind ] = min(trace(zci(iIntercept,1): zci(iIntercept + 1,1)));
	ind = ind + zci(iIntercept,1) - 1;
	valleys = cat(1,valleys,ind);
end


