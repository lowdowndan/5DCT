function report(aModel)

%% Load template
texTemplate = importdata(fullfile(getPath,'reporting','template.tex'));

%% Generate lines to add to template
lines = cell(2,1);
% Add patient information to header
lines{1} = sprintf('\\lhead{\\textbf{%s \\\\ %s, %s}\\\\ \\textbf{\\today}}',num2str(aModel.study.patient.id), aModel.study.patient.last,aModel.study.patient.first);

logoPath = fullfile(fiveDpath,'reporting','composite_logo');
%logoPath = fullfile(getPath,'reporting','ucla');

lines{2} = sprintf('\\rhead{\\includegraphics[width=.6\\textwidth]{"%s"}}', logoPath);


ins = '\begin{document}';
lines = cat(1,lines,ins);

% Heading
ins = '\section*{Motion Modeling}';
lines = cat(1,lines,ins);

ins = '\subsection*{Summary}';
lines = cat(1,lines,ins);

% Model Summary
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

ins = sprintf('Number of scans & %02d \\\\',length(aModel.runScans));
lines = cat(1,lines,ins);

ins = sprintf('Reference scan  & %02d \\\\',aModel.registration.refScan);
lines = cat(1,lines,ins);

ins = sprintf('Residual mean (mm) & %0.2f \\\\',aModel.residualStatistics.mean);
lines = cat(1,lines,ins);

ins = sprintf('Residual standard deviation (mm) & %0.2f \\\\',aModel.residualStatistics.std);
lines = cat(1,lines,ins);

ins = sprintf('Residual 95th percentile (mm) & %0.2f \\\\',aModel.residualStatistics.ninefive);
lines = cat(1,lines,ins);


ins = '\bottomrule';
lines = cat(1,lines,ins);

ins = '\end{tabular}';
lines = cat(1,lines,ins);

ins = '\end{table}';
lines = cat(1,lines,ins);

%ins = '\clearpage';
%lines = cat(1,lines,ins);
% Error histogram

ins = '\subsection*{Residual histogram}';
lines = cat(1,lines,ins);

ins = '\begin{figure}[h!]';
lines = cat(1,lines,ins);

ins = '\centering';
lines = cat(1,lines,ins);

ins = ['\includegraphics[width=0.75\textwidth]{' fullfile(aModel.folder,'documents', 'histogram.png') '}'];
lines = cat(1,lines,ins);

ins = '\end{figure}';
lines = cat(1,lines,ins);

ins = '\clearpage';
lines = cat(1,lines,ins);

% Coronal error MIP

ins = '\subsection*{Residual AP/Lat MIPs (mm)}';
lines = cat(1,lines,ins);

ins = '\begin{figure}[h!]';
lines = cat(1,lines,ins);

ins = '\centering';
lines = cat(1,lines,ins);

%ins = '\caption*{\large{\textbf{MIP of Model residual (mm)}}}';
%lines = cat(1,lines,ins);
ins = ['\includegraphics[width=0.75\textwidth]{' fullfile(aModel.folder,'documents', 'residual_cor.png') '}'];
lines = cat(1,lines,ins);

ins = '\end{figure}';
lines = cat(1,lines,ins);

% Sagittal error MIP
ins = '\begin{figure}[h!]';
lines = cat(1,lines,ins);

ins = '\centering';
lines = cat(1,lines,ins);

ins = ['\includegraphics[width=0.75\textwidth]{' fullfile(aModel.folder,'documents', 'residual_sag.png') '}'];
lines = cat(1,lines,ins);

ins = '\end{figure}';
lines = cat(1,lines,ins);


ins = '\clearpage';
lines = cat(1,lines,ins);


%% Reconstruction overlays

ins = '\subsection*{Original Scan Reconstructions}';
lines = cat(1,lines,ins);

for jScan = 1:length(aModel.runScans)

iScan = aModel.runScans(jScan);

ins = '\begin{figure}[h!]';
lines = cat(1,lines,ins);

ins = sprintf('\\caption*{Scan %02d}', iScan);
lines = cat(1,lines,ins);


ins = '\centering';
lines = cat(1,lines,ins);

ins = '\begin{subfigure}[b]{0.30\textwidth}';
lines = cat(1,lines,ins);
ins = ['\includegraphics[width=\textwidth]{' fullfile(aModel.folder,'documents',sprintf('recon_%02d_cor.png',iScan)) '}'];
lines = cat(1,lines,ins);
ins = '\end{subfigure}';
lines = cat(1,lines,ins);

ins = '\begin{subfigure}[b]{0.30\textwidth}';
lines = cat(1,lines,ins);
ins = ['\includegraphics[width=\textwidth]{' fullfile(aModel.folder,'documents',sprintf('recon_%02d_sag_l.png',iScan)) '}'];
lines = cat(1,lines,ins);
ins = '\end{subfigure}';
lines = cat(1,lines,ins);

ins = '\begin{subfigure}[b]{0.30\textwidth}';
lines = cat(1,lines,ins);
ins = ['\includegraphics[width=\textwidth]{' fullfile(aModel.folder,'documents',sprintf('recon_%02d_sag_r.png',iScan)) '}'];
lines = cat(1,lines,ins);
ins = '\end{subfigure}';
lines = cat(1,lines,ins);


ins = '\end{figure}';
lines = cat(1,lines,ins);
end

ins = '\clearpage';
lines = cat(1,lines,ins);




%% End document
ins = '\end{document}';
lines = cat(1,lines,ins);



%% Write to file
texOut = cat(1,texTemplate,lines);
chkmkdir(fullfile(aModel.folder,'report'));
fReport = fopen(fullfile(aModel.folder,'report','report.tex'), 'w');
fSpec = '%s\n';

for iRow = 1:size(texOut,1)
    fprintf(fReport,fSpec,texOut{iRow});
end
fclose(fReport);

%% Compile
startDir = pwd;
cd(fullfile(aModel.folder,'report'));
system(sprintf('texliveonfly %s >/dev/null', 'report.tex'));
cd(startDir);
