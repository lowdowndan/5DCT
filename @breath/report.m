function report(aBreath)

%% Load template
texTemplate = importdata(fullfile(getPath,'reporting','template.tex'));

%% Generate lines to add to template
lines = cell(2,1);
% Add patient information to header
lines{1} = sprintf('\\lhead{\\textbf{%s \\\\ %s, %s}\\\\ \\textbf{\\today}}',num2str(aBreath.study.patient.id), aBreath.study.patient.last,aBreath.study.patient.first);

logoPath = fullfile(fiveDpath,'reporting','composite_logo');
%logoPath = fullfile(getPath,'reporting','ucla');

lines{2} = sprintf('\\rhead{\\includegraphics[width=.6\\textwidth]{"%s"}}', logoPath);


ins = '\begin{document}';
lines = cat(1,lines,ins);


%% Representative breath
ins = '\section*{Respiratory surrogate analysis}';
lines = cat(1,lines,ins);



% Summary
ins = '\begin{table}[H]';
lines = cat(1,lines,ins);
ins = '\centering';
lines = cat(1,lines,ins);

ins = '\begin{tabular}{cc}';
lines = cat(1,lines,ins);

ins = '\toprule';
lines = cat(1,lines,ins);

ins = '\midrule';
lines = cat(1,lines,ins);

ins = sprintf('Included lower percentile & %02dth \\\\',aBreath.percentileInterval(1));
lines = cat(1,lines,ins);

ins = sprintf('Included upper percentile  & %02dth \\\\',aBreath.percentileInterval(2));
lines = cat(1,lines,ins);

ins = '\bottomrule';
lines = cat(1,lines,ins);

ins = '\end{tabular}';
lines = cat(1,lines,ins);

ins = '\end{table}';
lines = cat(1,lines,ins);

% Annotaed trace
ins = '\subsection*{Respiratory trace}';
lines = cat(1,lines,ins);

ins = '\begin{figure}[H]';
lines = cat(1,lines,ins);

ins = '\centering';
lines = cat(1,lines,ins);

ins = ['\includegraphics[width=0.95\textwidth]{' fullfile(aBreath.folder,'documents', 'trace_annotated.png') '}'];
lines = cat(1,lines,ins);

ins = '\end{figure}';
lines = cat(1,lines,ins);

ins = '\clearpage';
lines = cat(1,lines,ins);

% Histogram trace
ins = '\subsection*{Respiratory phase histogram}';
lines = cat(1,lines,ins);

ins = '\begin{figure}[H]';
lines = cat(1,lines,ins);

ins = '\centering';
lines = cat(1,lines,ins);

ins = ['\includegraphics[width=0.85\textwidth]{' fullfile(aBreath.folder,'documents', 'histogram.png') '}'];
lines = cat(1,lines,ins);

ins = '\end{figure}';
lines = cat(1,lines,ins);

% Extrema trace
ins = '\subsection*{Waveform segmentation}';
lines = cat(1,lines,ins);

ins = '\begin{figure}[H]';
lines = cat(1,lines,ins);

ins = '\centering';
lines = cat(1,lines,ins);

ins = ['\includegraphics[width=0.95\textwidth]{' fullfile(aBreath.folder,'documents', 'extrema.png') '}'];
lines = cat(1,lines,ins);

ins = '\end{figure}';
lines = cat(1,lines,ins);


ins = '\clearpage';
lines = cat(1,lines,ins);

% Representative breath
ins = '\subsection*{Representative breath}';
lines = cat(1,lines,ins);

ins = '\begin{figure}[h!]';
lines = cat(1,lines,ins);

ins = '\centering';
lines = cat(1,lines,ins);

ins = ['\includegraphics[width=0.85\textwidth]{' fullfile(aBreath.folder,'documents', 'representativeBreath.png') '}'];
lines = cat(1,lines,ins);

ins = '\end{figure}';
lines = cat(1,lines,ins);

% Representative breath
ins = '\subsection*{Context}';
lines = cat(1,lines,ins);

ins = '\begin{figure}[h!]';
lines = cat(1,lines,ins);

ins = '\centering';
lines = cat(1,lines,ins);

ins = ['\includegraphics[width=0.95\textwidth]{' fullfile(aBreath.folder,'documents', 'representativeBreathContext.png') '}'];
lines = cat(1,lines,ins);

ins = '\end{figure}';
lines = cat(1,lines,ins);


ins = '\clearpage';
lines = cat(1,lines,ins);

%% Reconstruction


%% End document
ins = '\end{document}';
lines = cat(1,lines,ins);



%% Write to file
texOut = cat(1,texTemplate,lines);
chkmkdir(fullfile(aBreath.folder,'report'));
fReport = fopen(fullfile(aBreath.folder,'report','report.tex'), 'w');
fSpec = '%s\n';

for iRow = 1:size(texOut,1)
    fprintf(fReport,fSpec,texOut{iRow});
end
fclose(fReport);

%% Compile
startDir = pwd;
cd(fullfile(aBreath.folder,'report'));
system(sprintf('texliveonfly %s >/dev/null', 'report.tex'));
cd(startDir);
