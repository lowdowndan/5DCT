function report(aRegistration)

%% Load template
texTemplate = importdata(fullfile(fiveDpath,'reporting','template.tex'));

%% Generate lines to add to template
lines = cell(2,1);
% Add patient information to header
lines{1} = sprintf('\\lhead{\\textbf{%s \\\\ %s, %s}\\\\ \\textbf{\\today}}',num2str(aRegistration.study.patient.id), aRegistration.study.patient.last,aRegistration.study.patient.first);

logoPath = fullfile(fiveDpath,'reporting','composite_logo');
%logoPath = fullfile(getPath,'reporting','ucla');

lines{2} = sprintf('\\rhead{\\includegraphics[width=.6\\textwidth]{"%s"}}', logoPath);


ins = '\begin{document}';
lines = cat(1,lines,ins);


%% DIR
ins = '\section*{Deformable image registration}';
lines = cat(1,lines,ins);


nonRef = setdiff([1:aRegistration.study.nScans], aRegistration.refScan);

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
ins = ['\includegraphics[width=\textwidth]{' fullfile(aRegistration.folder,'documents',sprintf('%02d_cor.png',iScan)) '}'];
lines = cat(1,lines,ins);
ins = '\end{subfigure}';
lines = cat(1,lines,ins);

ins = '\begin{subfigure}[b]{0.30\textwidth}';
lines = cat(1,lines,ins);
ins = ['\includegraphics[width=\textwidth]{' fullfile(aRegistration.folder,'documents',sprintf('%02d_sag_l.png',iScan)) '}'];
lines = cat(1,lines,ins);
ins = '\end{subfigure}';
lines = cat(1,lines,ins);

ins = '\begin{subfigure}[b]{0.30\textwidth}';
lines = cat(1,lines,ins);
ins = ['\includegraphics[width=\textwidth]{' fullfile(aRegistration.folder,'documents',sprintf('%02d_sag_r.png',iScan)) '}'];
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
chkmkdir(fullfile(aRegistration.folder,'report'));
fReport = fopen(fullfile(aRegistration.folder,'report','report.tex'), 'w');
fSpec = '%s\n';

for iRow = 1:size(texOut,1)
    fprintf(fReport,fSpec,texOut{iRow});
end
fclose(fReport);

%% Compile
startDir = pwd;
cd(fullfile(aRegistration.folder,'report'));
system(sprintf('texliveonfly %s >/dev/null', 'report.tex'));
cd(startDir);
