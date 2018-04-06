function original_scans(aModel)

aStudy = aModel.study;

outputFolder = fullfile(aModel.folder,'original');
chkmkdir(outputFolder);

% Progres bar
originalBar = waitbar(0,'Reconstructing original scans...');

%% Reconstruct scans

% Load parameters
[aX, aY, aZ] = aModel.get_alpha;
[bX, bY, bZ] = aModel.get_beta;
[cX, cY, cZ] = aModel.get_constant;    

%img = aModel.registration.get_average_image;
img = aModel.study.get_image(aModel.registration.refScan);

for jScan = 1:aModel.nScans
    
    iScan = aModel.runScans(jScan);
        
    v = aStudy.v(:,iScan);
    f = aStudy.f(:,iScan);
    
%     %disp('debug test');
%     % Pre-allocate
%     imgSim = zeros(aModel.study.dim,'single');
%     
%     % Buffer region to account for out-of-plane motion
%     buffer = 20;
%     tic
%     for iSlice = 1:aModel.study.dim(3)
%         
%         iSlice
%         % Slice at bottom of image?    
%         if iSlice < (buffer + 1)
%         
%         taX = aX(:,:,1:2*buffer + 1);
%         taY = aY(:,:,1:2*buffer + 1);
%         taZ = aZ(:,:,1:2*buffer + 1);
%         
%         tbX = bX(:,:,1:2*buffer + 1);
%         tbY = bY(:,:,1:2*buffer + 1);
%         tbZ = bZ(:,:,1:2*buffer + 1);
%         
%         tcX = cX(:,:,1:2*buffer + 1);
%         tcY = cY(:,:,1:2*buffer + 1);
%         tcZ = cZ(:,:,1:2*buffer + 1);
%         
%         timg = img(:,:,1:2*buffer + 1);
%         tslice = iSlice;
%      
%         % Slice at top of image?    
%         elseif (iSlice > aModel.study.dim(3) - buffer - 1)
%         
%         taX = aX(:,:,end - 2*buffer : end);
%         taY = aY(:,:,end - 2*buffer : end);
%         taZ = aZ(:,:,end - 2*buffer : end);
%         
%         tbX = bX(:,:,end - 2*buffer : end);
%         tbY = bY(:,:,end - 2*buffer : end);
%         tbZ = bZ(:,:,end - 2*buffer : end);
%         
%         tcX = cX(:,:,end - 2*buffer : end);
%         tcY = cY(:,:,end - 2*buffer : end);
%         tcZ = cZ(:,:,end - 2*buffer : end);
%         
%         timg = img(:,:,end - 2*buffer : end);  
%         tslice = (2*buffer) - (aModel.study.dim(3) - iSlice);
%         
%         % Normal slice
%         else 
%                     
%         taX = aX(:,:,iSlice - buffer : iSlice + buffer);
%         taY = aY(:,:,iSlice - buffer : iSlice + buffer);
%         taZ = aZ(:,:,iSlice - buffer : iSlice + buffer);
%         
%         tbX = bX(:,:,iSlice - buffer : iSlice + buffer);
%         tbY = bY(:,:,iSlice - buffer : iSlice + buffer);
%         tbZ = bZ(:,:,iSlice - buffer : iSlice + buffer);
%         
%         tcX = cX(:,:,iSlice - buffer : iSlice + buffer);
%         tcY = cY(:,:,iSlice - buffer : iSlice + buffer);
%         tcZ = cZ(:,:,iSlice - buffer : iSlice + buffer);
%         
%         timg = img(:,:,iSlice - buffer : iSlice + buffer);
%         tslice = buffer + 1;
%    
%         end
%         
%         tmp = aModel.generate_image(v(iSlice),f(iSlice),taX,taY,taZ,tbX,tbY,tbZ,tcX,tcY,tcZ,timg);
%         imgSim(:,:,iSlice) = tmp(:,:,tslice);
%     
%         end
%        
%     toc
    %imgSim = aModel.generate_image(v,f,aX,aY,aZ,bX,bY,bZ,cX,cY,cZ);
    imgSim = aModel.deform_image(img,v,f,aX,aY,aZ,bX,bY,bZ,cX,cY,cZ);
  
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
