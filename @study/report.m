function report(aStudy)

%% Load template
texTemplate = importdata(fullfile(getPath,'reporting','template.tex'));

%% Generate lines to add to template
lines = cell(2,1);
% Add patient information to header
lines{1} = sprintf('\\lhead{\\textbf{%s \\\\ %s, %s}\\\\ \\textbf{\\today}}',num2str(aStudy.patient.id), aStudy.patient.last,aStudy.patient.first);

logoPath = fullfile(getPath,'reporting','composite_logo');
%logoPath = fullfile(getPath,'reporting','ucla');

lines{2} = sprintf('\\rhead{\\includegraphics[width=.6\\textwidth]{"%s"}}', logoPath);


ins = '\begin{document}';
lines = cat(1,lines,ins);

% Heading
ins = '\section*{Image and Breathing Surrogate Acquisition}';
lines = cat(1,lines,ins);

ins = '\subsection*{LabVIEW Data}';
lines = cat(1,lines,ins);


% LabVIEW Channels

% Bellows
ins = '\begin{figure}[h!]';
lines = cat(1,lines,ins);

ins = '\caption*{Bellows}';
lines = cat(1,lines,ins);

ins = '\centering';
lines = cat(1,lines,ins);

ins = ['\includegraphics[width=\textwidth]{' fullfile(aStudy.folder,'documents','channel_voltage.png') '}'];
lines = cat(1,lines,ins);

ins = '\end{figure}';
lines = cat(1,lines,ins);


% Xray on
ins = '\begin{figure}[h!]';
lines = cat(1,lines,ins);

ins = '\caption*{X-Ray On}';
lines = cat(1,lines,ins);

ins = '\centering';
lines = cat(1,lines,ins);

ins = ['\includegraphics[width=\textwidth]{' fullfile(aStudy.folder,'documents','channel_xrayOn.png') '}'];
lines = cat(1,lines,ins);

ins = '\end{figure}';
lines = cat(1,lines,ins);

% EKG
ins = '\begin{figure}[h!]';
lines = cat(1,lines,ins);

ins = '\caption*{EKG}';
lines = cat(1,lines,ins);

ins = '\centering';
lines = cat(1,lines,ins);

ins = ['\includegraphics[width=\textwidth]{' fullfile(aStudy.folder,'documents','channel_ekg.png') '}'];
lines = cat(1,lines,ins);

ins = '\end{figure}';
lines = cat(1,lines,ins);


%
ins = '\clearpage';
lines = cat(1,lines,ins);


% Scan Stop/Start
ins = '\section*{Scan Start/Stop Times}';
lines = cat(1,lines,ins);

ins = '\begin{figure}[H]';
lines = cat(1,lines,ins);

ins = '\centering';
lines = cat(1,lines,ins);

ins = ['\includegraphics[width=\textwidth]{' fullfile(aStudy.folder,'documents','scanSegments.png') '}'];
lines = cat(1,lines,ins);

ins = '\end{figure}';
lines = cat(1,lines,ins);

ins = '\clearpage';
lines = cat(1,lines,ins);


% Included scans
ins = '\section*{Scan Ranges}';
lines = cat(1,lines,ins);
ins = '\begin{figure}[h!]';
lines = cat(1,lines,ins);

ins = '\centering';
lines = cat(1,lines,ins);

ins = ['\includegraphics[width=\textwidth]{' fullfile(aStudy.folder,'documents','scanSelection.png') '}'];
lines = cat(1,lines,ins);

ins = '\end{figure}';
lines = cat(1,lines,ins);


ins = '\clearpage';
lines = cat(1,lines,ins);

% Surrogate Linearity
ins = '\section*{Bellows/Abdominal Height Linearity}';
lines = cat(1,lines,ins);

ins = '\begin{figure}[h!]';
lines = cat(1,lines,ins);

ins = '\centering';
lines = cat(1,lines,ins);

ins = ['\includegraphics[width=.75\textwidth]{' fullfile(aStudy.folder,'documents','surrogatePlot.png') '}'];
lines = cat(1,lines,ins);

ins = '\end{figure}';
lines = cat(1,lines,ins);


ins = '\clearpage';
lines = cat(1,lines,ins);


for iScan = 1:aStudy.nScans

ins = '\begin{figure}[h!]';
lines = cat(1,lines,ins);

ins = sprintf('\\caption*{Scan %02d}', iScan);
lines = cat(1,lines,ins);

ins = '\centering';
lines = cat(1,lines,ins);
ins = ['\includegraphics[width=\textwidth]{' fullfile(aStudy.folder,'documents','surrogateCalibration',sprintf('%02d.png',iScan)) '}'];
lines = cat(1,lines,ins);
ins = '\end{figure}';
lines = cat(1,lines,ins);

end




%% End document
ins = '\end{document}';
lines = cat(1,lines,ins);



%% Write to file
texOut = cat(1,texTemplate,lines);
chkmkdir(fullfile(aStudy.folder,'report'));
fReport = fopen(fullfile(aStudy.folder,'report','report.tex'), 'w');
fSpec = '%s\n';

for iRow = 1:size(texOut,1)
    fprintf(fReport,fSpec,texOut{iRow});
end
fclose(fReport);

%% Compile
startDir = pwd;
cd(fullfile(aStudy.folder,'report'));
system(sprintf('texliveonfly %s >/dev/null', 'report.tex'));
cd(startDir);
