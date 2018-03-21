
function add_registration(aStudy,refScan)

% Append to existing studies (if any)
nRegistration = numel(aStudy.registration) + 1;

if(nRegistration == 1)
aStudy.registration = registration(aStudy, refScan);
else
aStudy.registration(nRegistration) = registration(aStudy, refScan);
end

aStudy.patient.save;
end



