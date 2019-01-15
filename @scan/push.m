%%PUSH Push a scan to MIM.
% scan.push(scanFolder) sends all DICOM files in scanFolder
% to the MIM server.

function push(scanFolder)

%% Verify that scans have been generated

%% MIM info
ip = '10.2.139.20';
port = '104';

%% Push
pushCmd = ['storescu --scan-directories ' ip ' ' port ' "' scanFolder '/"'];
[s,r] = system(pushCmd);
end
