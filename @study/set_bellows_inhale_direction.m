
function set_bellows_inhale_direction(aStudy, inhaleDirection)

%% Direction provided?

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
end

%% Modify study object to reflect changes
aStudy.bellowsInhaleDirection = inhaleDirection;

end
