%% load_parameters
function [pX, pY, pZ] = load_parameters(aModel,p)

dimensions = {'X','Y','Z'};
parameters = cell(3,1);

dim = aModel.study.dim;

for iDim = 1:3
    filename = fullfile(aModel.folder,[p dimensions{iDim} '.dat']);
    fParameters = fopen(filename,'r');
    parameters{iDim} = fread(fParameters,'single'); 
    parameters{iDim} = reshape(parameters{iDim}, dim(1), dim(2), dim(3));
    fclose(fParameters);
end

pX = parameters{1};
pY = parameters{2};
pZ = parameters{3};

end



