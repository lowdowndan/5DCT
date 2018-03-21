
function add_breath(aStudy)

% Append to existing studies (if any)
nBreath = numel(aStudy.breath) + 1;

if(nBreath == 1)
aStudy.breath = breath(aStudy);
else
aStudy.breath(nBreath) = breath(aStudy);
end

aStudy.patient.save;
end



