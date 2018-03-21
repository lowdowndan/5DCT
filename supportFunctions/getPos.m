function pos = getPos(texTemplate, n);

pos = strfind(texTemplate, sprintf('%% --%d.',n));
pos = cellfun(@any,pos);
pos = find(pos,1,'first');


end

