%% Workflow script

function workflow_fived(aPatient)

%% Add study
addpath('/home/doconnell/Dropbox/4DCT/pulmonarytoolkit-read-only/')

aPatient.add_study;
aPatient.study.synchronize;
aPatient.study.report;
    
%% Add breath
aPatient.study.add_breath;
aPatient.study.breath.set_representative(1);
aPatient.study.breath.report

  
%% Add registration
aPatient.study.add_registration(1);
aPatient.study.registration.register;
aPatient.study.registration.slice;
aPatient.study.registration.set_representative_slices;
aPatient.study.registration.plot_overlays;
aPatient.study.registration.get_average_image;
aPatient.study.registration.report;

        
%% Add model
aPatient.study.add_model(aPatient.study.registration);
tic
aPatient.study.model.fit;
toc
aPatient.study.model.plot_residual;
aPatient.study.model.plot_overlays;
aPatient.study.model.report;
        
%% Add sequence
aPatient.study.model.add_sequence(aPatient.study.registration);
aPatient.study.model.sequence.set_reconstruction_points;
aPatient.study.model.sequence.generate_scans;
aPatient.study.model.sequence.push;

compStr = sprintf('5DCT for Patient %s generated successfully.', num2str(aPatient.id));
disp(compStr)
end
            
        
        
        
    