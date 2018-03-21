%% compileReport: Generate final report and publish to pdf

function acquisitionReport(aStudy,reportData)

pMin = sprintf('%dth',reportData.pMin);
pMax = sprintf('%dth',reportData.pMax);

%% Load template
texTemplate = importdata(fullfile(getPath,'reporting','acquisitionTemplate.tex'));

%% Generate lines to add to template

% Add patient information to header
pos(1) = getPos(texTemplate,1);
lines{1} = sprintf('\\lhead{\\textbf{%s, %s; %s}\\\\ \\textbf{\\today}}',reportData.last,reportData.first,reportData.mrn);

pos(2) = getPos(texTemplate,2);
logoPath = fullfile(getPath,'reporting','ucla');
lines{2} = sprintf('\\rhead{\\includegraphics[width=.25\\textwidth]{"%s"}}', logoPath);

pos(3) = getPos(texTemplate, 3);
lines{3} = sprintf('The minimum inspiration image was generated using the \\textbf{%s} percentile breathing amplitude and the maximum inspiration image was generated using the \\textbf{%s} percentile.',pMin, pMax);

pos(4) = getPos(texTemplate, 4);
lines{4} = sprintf('Number of Scans & %d \\\\',aStudy.nScans);

pos(5) = getPos(texTemplate, 5);
if isfield(reportData,'pitch');
lines{5} = sprintf('Pitch & %0.2f \\\\',reportData.pitch);
else
	lines{5} = '';
end

pos(6) = getPos(texTemplate, 6);
lines{6} = sprintf('mAs/scan & %d \\\\',reportData.exposure);

pos(7) = getPos(texTemplate, 7);
lines{7} = sprintf('Scan Length (cm) & %0.2f \\\\', reportData.scanLength);

pos(8) = getPos(texTemplate, 8);
pos(9) = getPos(texTemplate, 9);
if isfield(reportData,'ctdivol')

	lines{8} = sprintf('Total CTDI\\textsubscript{vol} & %0.2f \\\\',reportData.ctdivol);
	lines{9} = sprintf('Estimated Total Imaging Dose (mGy) & %0.2f \\\\',reportData.eDose);
else
	lines{8} = '';
	lines{9} = '';
end

pos(10) = getPos(texTemplate,10);
lines{10} = sprintf('\\includegraphics[width=.925\\textwidth]{%s}', fullfile(aStudy.folder,'report','tracePlot'));
pos(11) = getPos(texTemplate,11);
lines{11} = sprintf('\\includegraphics[width=.925\\textwidth]{%s}', fullfile(aStudy.folder,'report','histogramPlot'));


pos(12) = getPos(texTemplate,12);
lines{12} = sprintf('1 - 4 & %0.1f\\%%  \\\\', reportData.t1_4);

pos(13) = getPos(texTemplate,13);
lines{13} = sprintf('\\textbf{5 - 85} & \\textbf{%0.1f\\%%}  \\\\', reportData.t5_85);


pos(14) = getPos(texTemplate,14);
lines{14} = sprintf('86 - 94 & %0.1f\\%%  \\\\', reportData.t86_94);

pos(15) = getPos(texTemplate,15);
lines{15} = sprintf('\\textbf{5 - 95} & \\textbf{%0.1f\\%%}  \\\\', reportData.t5_95);

pos(16) = getPos(texTemplate,16);
lines{16} = sprintf('96 - 100 & %0.1f\\%%  \\\\', reportData.t96_100);

%spacing
pos(17) = getPos(texTemplate,17);
lines{17} = '';

% Substitute updated lines into template
texTemplate(pos) = lines;

%% Write to file
fReport = fopen(fullfile(aStudy.folder,'report','report.tex'), 'w');
fSpec = '%s\n';

for iRow = 1:size(texTemplate,1)
    fprintf(fReport,fSpec,texTemplate{iRow});
end
fclose(fReport);

%% Compile
startDir = pwd;
cd(fullfile(aStudy.folder,'report'));
system(sprintf('texliveonfly %s >/dev/null', 'report.tex'));
cd(startDir);
