%LEAKDOWN import a LabVIEW .tdms file, and perform a linear
% regression to determine the relationship between time
% and bellows amplitude.  This is intended to be used to analyze
% a measurement made with the bellows held at a steady tension
% using a motion phantom for a period of a few minutes in order
% to determine the leakage rate of the apparatus.
%
% [leakRate, correlationCoefficient] = leakdown(bellowsDataFilename) returns the leakage rate
% in PSI Vacuum / .01 s, and the Pearson correlation coefficient for the 
% linear regression


function [leakRate, correlationCoefficient] = leakdown(bellowsDataFilename)

%% Import .tdms file
[bellowsData, channels] = study.convert_tdms(bellowsDataFilename);
bellows = bellowsData(:,channels.voltage);
time = bellowsData(:,channels.time);

%% Select data range
load('fivedcolor');
bellowsColor = fivedcolor.blue;

percTen = .1 * range(bellows);
selectionPlot = figure;
set(selectionPlot,'units','normalized','position', [0.1000    0.1000    0.8100    0.8100]);
ylim([min(bellows) - percTen, max(bellows) + percTen]);
hold on
plot(time,bellows, 'color', bellowsColor, 'linewidth',1);
xlabel('Time (s)');
ylabel('Bellows Vacuum Pressure (PSI)');
title('Select a range to analyze leakage.');
set(gca,'FontSize',20);
pointList = selectdata('sel','r','Verify','off','Pointer','crosshair');
hold off
close(selectionPlot);

% Use only data within selected range
dataRange(1) = min(pointList);
dataRange(2) = max(pointList);





% Voltage Plot
voltFig = figure;
voltFig.Units = 'normalized';
%voltFig.Visible = 'off';
voltFig.Position =  [0.0 0.0 .99 .99];
plot(time,bellows, 'color', bellowsColor, 'linewidth',1);
xlabel('Time (s)');
ylabel('Voltage (V)');
set(gca,'fontsize',20);
voltFig.Color = [1 1 1];
