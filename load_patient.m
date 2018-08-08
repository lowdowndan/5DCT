%% Load patient script

function load_patient(id)

if(numel(num2str(id)) < 3)
	% Zero pad
	idStr = sprintf('%03d',id);
elseif(isa(id,'char'))
    idStr = id;
else
	idStr = num2str(id);
end

load(fullfile(fiveDdata,idStr,'patient.mat'));

% Place in workspace
assignin('base','aPatient',aPatient);

end

