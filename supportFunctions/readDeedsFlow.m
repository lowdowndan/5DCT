function deedsDVF = readDeedsFlow(baseFilename)

flowU = load_nii([baseFilename '_flowu.nii']);
flowV = load_nii([baseFilename '_flowv.nii']);
flowW = load_nii([baseFilename '_floww.nii']);

deedsDVF = zeros([size(flowU.img,1) size(flowU.img,2) 3 size(flowU.img,3)],'single');
deedsDVF(:,:,1,:) = flowU.img;
deedsDVF(:,:,2,:) = flowV.img;
deedsDVF(:,:,3,:) = flowW.img;
    