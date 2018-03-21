function smoothVoltage(aStudy)

aStudy.data(:,aStudy.channels.voltage) = smooth_sg(aStudy.data(:,aStudy.channels.voltage));

% Call for save of patient object
%notify(aStudy.patient,'statusChange');
end
