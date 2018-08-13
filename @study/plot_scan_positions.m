function plot_scan_positions(aStudy,sliceFig, allZpositions,commonRefZpositions,sharedMin,sharedMax,validScans,tolerance,img)

minZ = min(cell2mat(allZpositions));
maxZ = max(cell2mat(allZpositions));

nScansValid = sum(validScans);
nScansTotal = length(allZpositions);


%% Display reference image projection
figure(sliceFig);
sliceFig.Units = 'normalized';
sliceFig.Position =  [0.0 0.0 .99 .99];

hImg = imagesc([0, nScansTotal + 1], [allZpositions{1}(1) allZpositions{1}(end)],img);
colormap gray
hold on;

set(gca,'YDir','normal');
set(gca,'xlim',[0 nScansTotal + 1]);
set(gca,'ylim',[minZ - 10 maxZ + 10]);
set(gca,'color','k')

%% Plot scan positions

% Color definition
cGray = [.666 .666 .666];
cOrange = [0.8750 0.25 0.1094];
cBlue = [0 0.3320 0.6484];

cBlue = [1 0 1];
cOrange = cBlue;
cGray = cBlue;

scanIterate = [1:nScansTotal];
scanIterate = scanIterate(validScans);

for jScan = 1 : nScansValid

	iScan = scanIterate(jScan);

	% Plot points below shared minimum
	subMinPoints = allZpositions{iScan}(allZpositions{iScan} < sharedMin);
	if(any(subMinPoints))

		plot(iScan,min(subMinPoints),'+', 'color', cGray);
		hold on;
		plot(iScan,max(subMinPoints),'+', 'color', cGray);
		hold on;
		hExclude = plot([iScan iScan],[min(subMinPoints) max(subMinPoints)],'--', 'color', cGray, 'linewidth', 2.0);
		hold on;
	end

	supMaxPoints = allZpositions{iScan}(allZpositions{iScan}  > sharedMax);
	if(any(supMaxPoints))

		plot(iScan,min(supMaxPoints),'+', 'color', cGray);
		hold on;
		plot(iScan,max(supMaxPoints),'+', 'color', cGray);
		hold on;
		hExclude = plot([iScan iScan],[min(supMaxPoints) max(supMaxPoints)],'--', 'color', cGray, 'linewidth', 2.0);
		hold on;
	end

	% Get all z positions within the common range
	sharedPoints = allZpositions{iScan}(allZpositions{iScan} > sharedMin & allZpositions{iScan} < sharedMax);

	% Determine if they are the same as the the reference image
	if(isequal(numel(sharedPoints),numel(commonRefZpositions)) & max(abs(bsxfun(@minus,sharedPoints,commonRefZpositions))) > tolerance)

		plot(iScan,min(sharedPoints),'s', 'color', cOrange, 'markerfacecolor', cOrange);
		hold on;
		plot(iScan,max(sharedPoints),'s', 'color', cOrange, 'markerfacecolor', cOrange);
		hold on;
		hInterpolate = plot([iScan iScan],[min(sharedPoints) max(sharedPoints)],'-.', 'color', cOrange, 'linewidth',2.0);
		hold on;
        
    elseif(isequal(numel(sharedPoints), 0))
        
        % Do nothing
        
	else

		plot(iScan,min(sharedPoints),'s', 'color', cBlue, 'markerfacecolor', cBlue);
		hold on;
		plot(iScan,max(sharedPoints),'s', 'color', cBlue, 'markerfacecolor', cBlue);
		hold on;
		hCoincident = plot([iScan iScan],[min(sharedPoints) max(sharedPoints)],'-', 'color', cBlue, 'linewidth',2.0);
		hold on;
	end
end

grid on
grid minor

xlim([0 nScansTotal + 1]);
ylim([minZ - 10 maxZ + 10]);
plot([1 max(scanIterate)],[sharedMin, sharedMin],'linewidth',1.5,'color',cBlue);
plot([1 max(scanIterate)],[sharedMax, sharedMax],'linewidth',1.5,'color',cBlue);
set(gca,'xtick',[1:nScansTotal],'fontsize',20)
xlabel('Scan number');
ylabel('Slice location (mm)')


if (exist('hInterpolate','var') && exist('hExclude','var') && exist('hCoincident','var'))
hLegend = legend([hCoincident, hInterpolate, hExclude], 'Coincident', 'Interpolated','Outside range','location','northeastoutside');

elseif (exist('hExclude','var') && ~exist('hInterpolate','var') && exist('hCoincident','var'))
hLegend = legend([hCoincident, hExclude], 'Coincident','Outside range','location','northeastoutside');

elseif (exist('hInterpolate','var') && ~exist('hExclude', 'var') && exist('hCoincident','var'))
hLegend = legend([hCoincident, hInterpolate], 'Coincident','Interpolated','location','northeastoutside');

elseif(~exist('hExclude','var') && exist('hInterpolate','var') && ~exist('hCoincident','var'))
hLegend = legend([hInterpolate],'Interpolated','location','northeastoutside');

elseif(exist('hExclude','var') && ~exist('hInterpolate','var') && ~exist('hCoincident','var'))
hLegend = legend([hExclude],'Excluded','location','northeastoutside');


else
hLegend = legend([hCoincident], 'Coincident','location','northeastoutside');

end
    
 
hLegend.Color = [1 1 1];
sliceFig.Color = [1 1 1];
drawnow;

