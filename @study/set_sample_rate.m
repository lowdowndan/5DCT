function set_sample_rate(aStudy)

channels = aStudy.channels;
sampleRate = aStudy.rawData(2,channels.time) - aStudy.rawData(1,channels.time);

validateattributes(sampleRate,{'numeric'},{'finite','nonnegative','real','nonzero','<',1, 'numel',1});

if ~isequal(sampleRate,.01)
    warning(sprintf('Typical sample rate is .010s.  Current sample rate is %03d s.', sampleRate));
end

aStudy.sampleRate = sampleRate;
end
