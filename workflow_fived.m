%% Workflow script

function workflow_fived(aPatient)

%% Add study
aPatient.add_study;
aPatient.study.synchronize;
    
%% Add breath
aPatient.study.add_breath;
aPatient.study.breath.set_representative(1);

%% Import scans
aPatient.study.import_scans
  
%% Add registration
aPatient.study.add_registration(1);

tic;
aPatient.study.registration.register;
registerTime = toc;
disp(sprintf('Image registration took %.02f seconds.',registerTime));

aPatient.study.registration.slice;
aPatient.study.registration.set_representative_slices;
aPatient.study.registration.plot_overlays;
aPatient.study.registration.get_average_image;

        
%% Add model
aPatient.study.add_model(aPatient.study.registration);
aPatient.save;
tic
aPatient.study.model.fit;
modelTime = toc;
disp(sprintf('Model fitting took %.02f seconds.',modelTime));

aPatient.study.model.plot_residual;
aPatient.study.model.original_scans;
aPatient.study.model.plot_overlays;
        


%% Add sequence
aPatient.study.model.add_sequence(aPatient.study.breath);
aPatient.study.model.sequence.set_reconstruction_points;
aPatient.study.model.sequence.generate_scans;
%aPatient.study.model.sequence.push;

% Report
%aPatient.study.report;
%aPatient.study.breath.report
%aPatient.study.registration.report;
%aPatient.study.model.report;
%aPatient.study.sequence.report;

compStr = sprintf('5DCT for Patient %s generated successfully.', num2str(aPatient.id));
disp(compStr)
end
            
        
        
        
    
