function residual = get_mean_residual(aModel)

filename = fullfile(aModel.folder,'residual.dat');
fResidual = fopen(filename,'r');
residual = fread(fResidual,'single');
residual = reshape(residual,aModel.study.dim);
fclose(fResidual);

end






