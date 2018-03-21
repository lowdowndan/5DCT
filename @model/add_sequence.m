
function add_sequence(aModel,aBreath)

% Append to existing studies (if any)
nSequence = numel(aModel.sequence) + 1;

if(nSequence == 1)
aModel.sequence = sequence(aModel, aBreath);
else
aModel.sequence(nSequence) = sequence(aModel, aBreath);

aModel.study.patient.save;
end



