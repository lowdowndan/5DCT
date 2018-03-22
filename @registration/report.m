function report(aModel)

%% Load template
texTemplate = importdata(fullfile(getPath,'reporting','template.tex'));

%% Generate lines to add to template
lines = cell(2,1);
% Add patient information to header
lines{1} = sprintf('\\lhead{\\textbf{%s, %s; %s}\\\\ \\textbf{\\today}}',aModel.patient.last,aModel.patient.first,aModel.patient.mrn);

logoPath = fullfile(getPath,'reporting','ucla');
lines{2} = sprintf('\\rhead{\\includegraphics[width=.25\\textwidth]{"%s"}}', logoPath);


ins = '\begin{document}';
lines = cat(1,lines,ins);


%% DIR
ins = '\section*{5DCT DIR QA Report}';
lines = cat(1,lines,ins);


nonRef = setdiff(aModel.runScans, aModel.refScan);

for jScan = 1:length(nonRef);

iScan = nonRef(jScan);

ins = '\begin{figure}[h!]';
lines = cat(1,lines,ins);

ins = sprintf('\\caption*{Scan %02d}', iScan);
lines = cat(1,lines,ins);


ins = '\centering';
lines = cat(1,lines,ins);

ins = '\begin{subfigure}[b]{0.30\textwidth}';
lines = cat(1,lines,ins);
ins = ['\includegraphics[width=\textwidth]{' fullfile(aModel.folder,'report',sprintf('%02d_cor.png',iScan)) '}'];
lines = cat(1,lines,ins);
ins = '\end{subfigure}';
lines = cat(1,lines,ins);

ins = '\begin{subfigure}[b]{0.30\textwidth}';
lines = cat(1,lines,ins);
ins = ['\includegraphics[width=\textwidth]{' fullfile(aModel.folder,'report',sprintf('%02d_sag_l.png',iScan)) '}'];
lines = cat(1,lines,ins);
ins = '\end{subfigure}';
lines = cat(1,lines,ins);

ins = '\begin{subfigure}[b]{0.30\textwidth}';
lines = cat(1,lines,ins);
ins = ['\includegraphics[width=\textwidth]{' fullfile(aModel.folder,'report',sprintf('%02d_sag_r.png',iScan)) '}'];
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
