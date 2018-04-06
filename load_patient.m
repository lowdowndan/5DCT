%% Load patient script

function aPatient = load_patient(id)

if(numel(num2str(id)) < 3)
	% Zero pad
	idStr = sprintf('%03d',id);
else
	idStr = num2str(id);
end

load(fullfile(fiveDdata,idStr,'patient.mat'));

end

