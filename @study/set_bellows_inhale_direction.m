
function set_bellows_inhale_direction(aStudy, inhaleDirection)

%% Direction provided?

load('fivedcolor')
if exist('inhaleDirection','var')
validateattributes(inhaleDirection,{'logical','numeric'},{'finite','real','integer','numel',1,});

else
    
% Plot channels

dirSelect = figure;
set(dirSelect,'units','normalized','position', [0.1000    0.1000    0.8100    0.8100]);
hold on
set(gca,'fontsize',20);


load('fivedcolor');
bellowsColor = fivedcolor.blue;


plot(aStudy.data(:,aStudy.channels.voltage), 'color', bellowsColor);
set(gca,'xtick',[]);
ylabel('Volts (V)');
title('Enter 0 if inhalation is negative, 1 if inhalation is positive.');


options.WindowStyle = 'normal';    
inhaleDirection = inputdlg({'Enter 0 if inhalation is negative, 1 if positive.'}, 'Set bellow signal direction', [1 40],{''},options);
inhaleDirection = str2num(inhaleDirection{1});
validateattributes(inhaleDirection,{'logical','numeric'},{'finite','real','integer','numel',1,});
close(dirSelect);
end


%% Flip necessary?

if(isequal(inhaleDirection,0))
    
vRaw = aStudy.data(:,aStudy.channels.voltage);
vFlipped = max(vRaw) - vRaw;
aStudy.data(:,aStudy.channels.voltage) = vFlipped;

% Overwrite plot for report
    
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

end

%% Modify study object to reflect changes
aStudy.bellowsInhaleDirection = inhaleDirection;

end
