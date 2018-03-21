
function add_model(aStudy, aRegistration, runScans)

% Are run scans specified?  If not include all
if(~exist('runScans','var'))
    runScans = 1:aStudy.nScans;
end

% Append to existing studies (if any)
nModel = numel(aStudy.model) + 1;

if(nModel == 1)
aStudy.model = model(aStudy, aRegistration, runScans);

else
aStudy.model(nModel) = model(aStudy, aRegistration, runScans);
end

aStudy.patient.save;
end



