%% Convert dicom acquisition time value to seconds after midnight
function time = time2sec(dicomTime)

dicomTime = num2str(dicomTime);

% Missing leading 0 or trailing 0s?
if numel(dicomTime) < 13
    dotIndex = strfind(dicomTime,'.');
    warning('DICOM AcquisitionTime tag is missing digits.')
    
    if dotIndex < 7
        dicomTime = cat(2,'0',dicomTime);
    end
    
    if numel(dicomTime) == 12
        dicomTime = cat(2,dicomTime, '0');
        
    elseif numel(dicomTime) < 13
        nZeros = 13 - numel(dicomTime);
        dicomTime = cat(2,dicomTime, sprintf(['%0' num2str(nZeros) 's'], 0));
    end
end
    
    
time = datetime(dicomTime,'InputFormat','HHmmss.SSSSSS');
    
h = hour(time);
m = minute(time);
s = second(time);

time = (h * 60 * 60) + (m * 60) + s;
end
