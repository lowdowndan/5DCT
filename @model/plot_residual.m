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
lungOnly = mR(mR > 0);
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


% Load colormap
%load('greenErrorColormap');
load('rdylgn');

% Coronal plot

% Plot
residualFig = figure('visible','off');
hImg = imagesc(imrotate(corMip,90));
set(residualFig,'units','normalized');
set(residualFig,'Position',[0.0         0    0.99    0.99]);
colormap(rdylgn);
axis image;
caxis([0 5]);
set(gca,'xticklabel',[],'yticklabel',[]);
imgAlpha = zeros(size(corMip));
imgAlpha(logical(corMip)) = 1;
hImg.AlphaData = imrotate(imgAlpha,90);
set(gca,'color',[0 0 0])
box on
set(gca,'fontname','DroidSans','fontsize',40);
colorbar;
set(gcf,'color',[1 1 1]);

% Save
f = getframe(residualFig);
chkmkdir(fullfile(aModel.folder,'documents'));
imwrite(f.cdata,fullfile(aModel.folder,'documents','residual_cor.png'),'png');
close(residualFig);

% Sagittal plot

% Plot
residualFig = figure('visible','off');
hImg = imagesc(imrotate(sagMip,90));
set(residualFig,'units','normalized');
set(residualFig,'Position',[0.0         0    0.99    0.99]);
colormap(rdylgn);
axis image;
caxis([0 5]);
set(gca,'xticklabel',[],'yticklabel',[]);
imgAlpha = zeros(size(sagMip));
imgAlpha(logical(sagMip)) = 1;
hImg.AlphaData = imrotate(imgAlpha,90);
set(gca,'color',[0 0 0])
box on
set(gca,'fontname','DroidSans','fontsize',40);
colorbar;
set(gcf,'color',[1 1 1]);

% Save
f = getframe(residualFig);
chkmkdir(fullfile(aModel.folder,'documents'));
imwrite(f.cdata,fullfile(aModel.folder,'documents','residual_sag.png'),'png');
close(residualFig);


%% Histogram
edges = [0:5, inf];
counts = histc(mR(logical(mask)),edges);
counts(5) = counts(5) + counts(5);
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
xlabel('Model residual (mm)')
ylabel('% Lung voxels')

f = getframe(histFig);
imwrite(f.cdata,fullfile(aModel.folder,'documents','histogram.png'),'png');
close(histFig);

