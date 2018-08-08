function isTDMS = is_tdms(filename)


%% File exists?

if (~exist(filename,'file'))
	error('File ''%s'' not found.', filename);
end

%% File can be opened?

fid = fopen(filename);

if (isequal(fid,-1))
	error('File ''%s'' could not be opened.', filename);
end

%% Is it a .tdms file?

% From convertTDMS by Brad Humphreys
% TDSm should be the first characters in a .tdms file

Ttag = fread(fid,1,'uint8','l');
Dtag=fread(fid,1,'uint8','l');
Stag=fread(fid,1,'uint8','l');
mtag=fread(fid,1,'uint8','l');

isTDMS = (Ttag==84 && Dtag==68 && Stag==83 && mtag==109);
