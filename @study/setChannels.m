%% getChannels
% Select the channels for x-ray on, bellows, and ekg data

function setChannels(aStudy, channels, noPlot)


if ~exist('noPlot','var')
	noPlot = false;
end

%% Validate noPlot
validateattributes(noPlot,{'logical','numeric'},{'finite','real','integer','numel',1,});


if noPlot

% Don't plot, make sure channels argument is passed
assert(logical(exist('channels','var')), 'channels structure must be input if plotting is suppressed.');

else

%% Plot channels
chanSelect = figure;
load('fivedcolor')
set(chanSelect,'units','normalized','position', [0.1000    0.1000    0.8100    0.8100]);
hold on
set(gca,'fontsize',20);
plot(aStudy.data(:,2), 'color', fivedcolor.blue, 'linewidth',1);
plot(aStudy.data(:,3), 'color', fivedcolor.orange, 'linewidth',1);
plot(aStudy.data(:,4), 'color', fivedcolor.red, 'linewidth',1);
plot(aStudy.data(:,5), 'color', fivedcolor.black, 'linewidth',1);

% plot(aStudy.data(:,3), 'g', 'linewidth',1);
% plot(aStudy.data(:,4), 'b', 'linewidth',1);
% plot(aStudy.data(:,5), 'k', 'linewidth',1);
set(gca,'xtick',[]);
ylabel('Volts (V)');

legend('2','3','4','5');

%% Prompt for channels
if nargin < 2
      
	title('Enter channel numbers.');
	
	% Get channels 
	channels = struct;
	channels.time = 1;

    
    userResp = inputdlg({'Bellows','X-Ray On', 'EKG'}, 'Enter channel numbers.', [1 30]);
    channels.voltage = str2num(userResp{1});
	channels.xrayOn = str2num(userResp{2});
	channels.ekg = str2num(userResp{3});
end
end

%% Validate input
assert((isnumeric(channels.voltage) && 1 < channels.voltage && channels.voltage <= 5),'FiveD:InvalidDataChannel','Invalid voltage channel number.');
assert((isnumeric(channels.xrayOn) && 1 < channels.xrayOn && channels.xrayOn <= 5 && (channels.xrayOn ~= channels.voltage)),'FiveD:InvalidDataChannel','Invalid xrayOn channel number.');
assert((isnumeric(channels.ekg) && 1 < channels.ekg && channels.ekg <= 5),'FiveD:InvalidDataChannel','Invalid ekg channel number.');

%% Validate input
assert(isa(channels,'struct'));
assert(isfield(channels,'time'));
assert(isfield(channels,'voltage'));
assert(isfield(channels,'xrayOn'));
assert(isfield(channels,'ekg'));

channelNames = fieldnames(channels);
assert(length(channelNames) == 4);

% Set channel data
aStudy.channels = channels;

if noPlot

	% Do nothing
else

	% Save channel data
	chanLabels = sprintf('Bellows: %d; X-Ray On: %d; EKG: %d',channels.voltage,channels.xrayOn,channels.ekg);
	xlabel(chanLabels);
	title('');
	chkmkdir(fullfile(aStudy.folder,'documents'));
    print(fullfile(aStudy.folder,'documents','channels.png'),'-dpng');
    %close(chanSelect);
    
    % Seperate plots
    
    % Voltage Plot
    bellowsColor = fivedcolor.blue;
    voltFig = figure;
    voltFig.Units = 'normalized';
    voltFig.Visible = 'off';
    voltFig.Position =  [0.0 0.0 .99 .99];
    plot(aStudy.data(:,1),aStudy.data(:,aStudy.channels.voltage), 'color', bellowsColor, 'linewidth',1);
    xlabel('Time (s)');
    ylabel('Voltage (V)');
    set(gca,'fontsize',20);
    voltFig.Color = [1 1 1];
    
    % Save
    f = getframe(gcf);
    imwrite(f.cdata,fullfile(aStudy.folder,'documents','channel_voltage.png'),'png');
    
    close(voltFig);
    
    
    % X-ray on
    xrayColor = fivedcolor.black;   
    xrayFig = figure;
    xrayFig.Units = 'normalized';
    xrayFig.Visible = 'off';
    xrayFig.Position =  [0.0 0.0 .99 .99];
    plot(aStudy.data(:,1),aStudy.data(:,aStudy.channels.xrayOn), 'color', xrayColor, 'linewidth',1);
    xlabel('Time (s)');
    ylabel('Voltage (V)');
    set(gca,'fontsize',20);
    xrayFig.Color = [1 1 1];
    
    % Save
    f = getframe(gcf);
    imwrite(f.cdata,fullfile(aStudy.folder,'documents','channel_xrayOn.png'),'png');
    close(xrayFig);
        
    % EKG
    ekgColor = fivedcolor.orange;
    ekgFig = figure;
    ekgFig.Units = 'normalized';
    ekgFig.Visible = 'off';
    ekgFig.Position =  [0.0 0.0 .99 .99];
    plot(aStudy.data(:,1),aStudy.data(:,aStudy.channels.ekg), 'color', ekgColor, 'linewidth',1);
    xlabel('Time (s)');
    ylabel('Voltage (V)');
    set(gca,'fontsize',20);
    ekgFig.Color = [1 1 1];
    
    % Save
    f = getframe(gcf);
    imwrite(f.cdata,fullfile(aStudy.folder,'documents','channel_ekg.png'),'png');
    close(ekgFig);
end

% Call for save of patient object
aStudy.patient.save;
end
