% Parse output of dcmdump (part of DCMTK).  If a file is not valid, return
% nan.

function dcmTableRow = parseDump(filename, dcmdumpResult)

nColumns = 9;
% Convert to cell array
dcmdumpResult = strsplit(dcmdumpResult,'\n')';
dcmdumpResult = dcmdumpResult(:);

% Remove empty entry
if isempty(dcmdumpResult{end})
    dcmdumpResult(end) = [];
end


% Remove duplicates
tags = cellfun(@(x) x(1:11), dcmdumpResult, 'uni', 0);
tags = tags(:);

[~,inds] = unique(tags,'last');
dcmdumpResult = dcmdumpResult(inds);
dcmdumpResult = sort(dcmdumpResult);

% Pre-allocate output
dcmTableRow = cell(1,nColumns);
pixelSpacing = zeros(2,1);
imgDim = zeros(2,1);
rescale = zeros(2,1);
dcmTableRow{1} = filename;

% Get positions of brackets surrounding data
dataStart = strfind(dcmdumpResult, '[');
dataStart = cell2mat(dataStart) + 1;



dataEnd = strfind(dcmdumpResult, ']'); 
dataEnd = cell2mat(dataEnd) - 1;

%Handle topogram
if length(dataStart) < 8
    warning(sprintf('File %s is likely a topogram.  Ignoring.',filename));
    %dcmTableRow{2} = dcmdumpResult{3}(dataStart(2):dataEnd(2));
    dcmTableRow{1} = nan;
    return;
end

parenEnd = strfind(dcmdumpResult, ')');
%parenEnd{3} = parenEnd{2}(1);
parenEnd{2} = parenEnd{2}(1);
parenEnd = cell2mat(parenEnd);


% Series UID
seriesRow = 4;
dcmTableRow{2} = dcmdumpResult{seriesRow}(dataStart(seriesRow):dataEnd(seriesRow));

% Acquisition Time
timeRow = 2;
dcmTableRow{3} = (dcmdumpResult{timeRow}(dataStart(timeRow):dataEnd(timeRow)));



% Z position (keep whole image position patient tag)
zRow = 5;
dcmTableRow{4} = dcmdumpResult{zRow}(dataStart(zRow):dataEnd(zRow));
coordinateDelims = strfind(dcmTableRow{4},'\');
dcmTableRow{4} = [ str2num(dcmTableRow{4}(1: coordinateDelims(1) - 1));
                   str2num(dcmTableRow{4}(coordinateDelims(1) + 1: coordinateDelims(2) - 1));
                   str2num(dcmTableRow{4}(coordinateDelims(2) + 1:end)) ];

% Slice Thickness
tRow = 3;
dcmTableRow{5} = str2num(dcmdumpResult{tRow}(dataStart(tRow):dataEnd(tRow)));


% [X Spacing; Y Spacing]
sRow = 8;
dcmTableRow{6} = dcmdumpResult{sRow}(dataStart(sRow - 2):dataEnd(sRow - 2));
coordinateDelims = strfind(dcmTableRow{6},'\');
pixelSpacing(1) = str2num(dcmTableRow{6}(1:coordinateDelims(1) - 1));
pixelSpacing(2) = str2num(dcmTableRow{6}(coordinateDelims(1) + 1 : end));
dcmTableRow{6} = pixelSpacing;


imRow = 6;
imRow1 = 6;
imRow2 = 7;

imgDim(1) = str2num(dcmdumpResult{imRow}(parenEnd(imRow1) + 5:parenEnd(imRow1) + 8));
imgDim(2) = str2num(dcmdumpResult{imRow}(parenEnd(imRow2) + 5:parenEnd(imRow2) + 8));

dcmTableRow{7} = imgDim;


rRow1 = 10;
rRow2 = 9;
rescale(1) = str2num(dcmdumpResult{rRow1}(dataStart(rRow1 - 2):dataEnd(rRow1 - 2)));
rescale(2) = str2num(dcmdumpResult{rRow2}(dataStart(rRow2 - 2):dataEnd(rRow2 - 2)));
dcmTableRow{8} = rescale;

sliceRow = 1;
dcmTableRow{9} = dcmdumpResult{sliceRow}(dataStart(sliceRow):dataEnd(sliceRow));

end
