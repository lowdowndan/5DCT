function original_scans(aModel)

aStudy = aModel.study;

outputFolder = fullfile(aModel.folder,'original');
mkdir(outputFolder);

% Progres bar
originalBar = waitbar(0,'Reconstructing original scans...');

%% Reconstruct scans

% Load parameters
[aX, aY, aZ] = aModel.get_alpha;
[bX, bY, bZ] = aModel.get_beta;
[cX, cY, cZ] = aModel.get_constant;    

for jScan = 1:aModel.nScans
    
    iScan = aModel.runScans(jScan);
        
    v = aStudy.v(:,iScan);
    f = aStudy.f(:,iScan);
    
    imgSim = aModel.generate_image(v,f,aX,aY,aZ,bX,bY,bZ,cX,cY,cZ);
  
    % Convert to nii
    scanFolder = fullfile(outputFolder,sprintf('%02d',iScan));
    imgNii = make_nii(imgSim, [1 1 1]);
       
    % Save
    mkdir(scanFolder);
    save_nii(imgNii,fullfile(scanFolder, 'simulated.nii'));
       
     try
     waitbar(iScan / aModel.nScans, originalBar);
     end
       
       
end
    
    try
        close(originalBar)
    end
