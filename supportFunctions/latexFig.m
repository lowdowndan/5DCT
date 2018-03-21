function texLines = latexFig(filename,width)

texLines = cell(4,1);
texLines{1} = '\begin{figure}[H]';
texLines{2} = '\centering';
texLines{3} = sprintf('\\includegraphics[width=%.2f\\textwidth]{%s}', width,filename);
texLines{4} = '\end{figure}';