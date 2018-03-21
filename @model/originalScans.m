function originalScans(aModel, img)

aStudy = aModel.study;

% % Registration w/ deeds
% path = getenv('PATH');   
% path = [path ':/usr/local/deedsMIND'];
% setenv('PATH',path)

outputFolder = fullfile(aModel.folder,'original');
mkdir(outputFolder);

% Set up grid
ii = 1:aModel.dim(1);
jj = 1:aModel.dim(2);
kk = 1:aModel.dim(3);
   
[~,~,Z] = meshgrid(ii,jj,kk);
Z = single(Z);   

%% Reference image
if~exist('img','var')
img = single(aModel.img);
end


% Progres bar
originalBar = waitbar(0,'Reconstructing original scans...');

%% Reconstruct scans

for iScan = 1:aStudy.nScans;      
        
       v = aStudy.v(:,iScan);
       f = aStudy.f(:,iScan);
      
       % Matrix of v values of target image
       vMat = permute(repmat(v, [1, aModel.dim(1), aModel.dim(2)]),[2,3,1]);
       
       % Matrix of f values of target image
       fMat = permute(repmat(f, [1, aModel.dim(1), aModel.dim(2)]),[2,3,1]);
      
       % Deform reference image
       imgDeformed = aModel.deformImage(img,vMat,fMat);
      
       % Convert to nii
       scanFolder = fullfile(outputFolder,sprintf('%02d',iScan));
       imgNii = make_nii(imgDeformed, [1 1 1]);
       
       % Save
       mkdir(scanFolder);
       save_nii(imgNii,fullfile(scanFolder, 'reconstructedScan.nii'));
       
       try
           waitbar(iScan / aStudy.nScans, originalBar);
       end
       
       
end
    
    try
        close(originalBar)
    end
