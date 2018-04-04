%% plot_residual
%
% Display model residual

function plot_residual(aModel)

%% Verify that model has been fit
if ~(exist(fullfile(aModel.folder,'alphaX.dat'),'file'))
    error('Model parameters not found.  Run fit method.');
end

%% Get residual, mask and plot

% Load mean residual
mR = aModel.get_mean_residual;

% Load mask
mask = aModel.study.get_mask(aModel.registration.refScan);

% Mask
mR = mR .* mask;

% Stats
lungOnly = mR(logical(mask));
lungOnly = lungOnly(:);
aModel.residualStatistics.mean = mean(lungOnly);
aModel.residualStatistics.std = std(lungOnly);
aModel.residualStatistics.ninefive = prctile(lungOnly,95);

% Get slices
%corMask = study.get_slice(mask,1,corSlice);
%sagMask = study.get_slice(mask,2,sagSlice);

% MIPs
corMip = squeeze(max(mR,[],1));
sagMip = squeeze(max(mR,[],2));

corMip = imrotate(corMip,90);
sagMip = imrotate(sagMip,90);

% Load reference image
img = aModel.study.get_image(aModel.registration.refScan);
imgCorProj = mat2gray(squeeze(sum(img,1)));
imgSagProj = mat2gray(squeeze(sum(img,2)));

imgCorProj = imrotate(imgCorProj,90);
imgSagProj = imrotate(imgSagProj,90);

% Convert to RGB
imgCorProjRGB = cat(3,imgCorProj,imgCorProj,imgCorProj);
imgSagProjRGB = cat(3,imgSagProj,imgSagProj,imgSagProj);


% Load colormap
%load('greenErrorColormap');
load('rdylgn');
load('colorbar_data');

%% Coronal plot

% HARDCODED COLOR RANGE: 0 to 5 mm
% Scale
minres = 0;
maxres = 5;
corMip(corMip > maxres) = maxres;
ncolor = size(rdylgn,1);
scaled = round(1 + (ncolor - 1) * (corMip - minres) / (maxres - minres));
corMipRGB = ind2rgb(scaled,rdylgn);

% Scale colorbar (size)
cbar = single(cbar);
cbarScaled = imresize(cbar, [size(corMipRGB,1) nan]);
cbarScaled = cbarScaled / 255;

% Expand 
widthCbar = size(cbarScaled,2);
corMipRGB = cat(2,corMipRGB,ones(size(corMipRGB,1),widthCbar,3));
imgCorProjRGB = cat(2,imgCorProjRGB,ones(size(corMipRGB,1),widthCbar,3));

% Place colorbar on both layers
%corMipRGB(:,end-widthCbar + 1:end,:) = cbarScaled;
imgCorProjRGB(:,end-widthCbar + 1:end,:) = cbarScaled;


% Plot
residualFig = figure('visible','off');

% Background
hBg = imshow(imgCorProjRGB);
axis image;

% Foreground
hold on;
hRes = imshow(corMipRGB);
axis image

% Transparency
imgAlpha = zeros(size(corMipRGB,1), size(corMipRGB,2));
imgAlpha(logical(cat(2,corMip,zeros(size(corMip,1),widthCbar)))) = 1;
hRes.AlphaData = imgAlpha;
box on
set(residualFig,'Color',[1 1 1]);

% Save
f = getframe(residualFig);
chkmkdir(fullfile(aModel.folder,'documents'));
imwrite(f.cdata,fullfile(aModel.folder,'documents','residual_cor.png'),'png');
close(residualFig);

%% Sagittal plot

% HARDCODED COLOR RANGE: 0 to 5 mm
% Scale
minres = 0;
maxres = 5;
sagMip(sagMip > maxres) = maxres;
ncolor = size(rdylgn,1);
scaled = round(1 + (ncolor - 1) * (sagMip - minres) / (maxres - minres));
sagMipRGB = ind2rgb(scaled,rdylgn);

% Scale colorbar (size)
cbar = single(cbar);
cbarScaled = imresize(cbar, [size(sagMipRGB,1) nan]);
cbarScaled = cbarScaled / 255;

% Expand 
widthCbar = size(cbarScaled,2);
sagMipRGB = cat(2,sagMipRGB,ones(size(sagMipRGB,1),widthCbar,3));
imgSagProjRGB = cat(2,imgSagProjRGB,ones(size(sagMipRGB,1),widthCbar,3));

% Place colorbar on both layers
imgSagProjRGB(:,end-widthCbar + 1:end,:) = cbarScaled;


% Plot
residualFig = figure('visible','off');

% Background
hBg = imshow(imgSagProjRGB);
axis image;

% Foreground
hold on;
hRes = imshow(sagMipRGB);
axis image

% Transparency
imgAlpha = zeros(size(sagMipRGB,1), size(sagMipRGB,2));
imgAlpha(logical(cat(2,sagMip,zeros(size(sagMip,1),widthCbar)))) = 1;
hRes.AlphaData = imgAlpha;
box on
set(residualFig,'Color',[1 1 1]);

% Save
f = getframe(residualFig);
chkmkdir(fullfile(aModel.folder,'documents'));
imwrite(f.cdata,fullfile(aModel.folder,'documents','residual_sag.png'),'png');
close(residualFig);


%% Histogram
edges = [0:5, inf];
counts = histc(mR(logical(mask)),edges);
%counts(5) = counts(5) + counts(6);
counts(6) = [];

counts = counts ./ sum(counts(:));
counts = counts * 100;
histFig = figure('visible','off');
histbar = bar(counts,'hist');
%histFig.Visible = 'off';
ylim([0 100]);
colormap(rdylgn);
set(gca,'fontname','droidsans','fontsize',40);
set(gcf,'Units','normalized');
set(gcf,'Color',[1 1 1]);
set(gcf,'OuterPosition', [0         0    0.99    0.99]);
set(gca,'xticklabel',{'0','1','2','3','4','5 +'})
xlabel('Model residual (mm)')
ylabel('% Lung voxels')

f = getframe(histFig);
imwrite(f.cdata,fullfile(aModel.folder,'documents','histogram.png'),'png');
close(histFig);

