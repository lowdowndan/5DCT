%% Workflow script

new_patient;

    % Add patient
    aPatient = patient(id);
    
    % Add study
    aPatient.add_study;
    aPatient.study.synchronize;
    aPatient.study.report;
    
    % Add breath
    aPatient.study.add_breath;
    aPatient.study.breath.set_representative;
    aPatient.study.breath.report
    % Default to semi-auto mode?
    
    % Add registration
    aPatient.study.add_registration;
        aPatient.study.registration.register;
        aPatient.study.registration.slice;
        aPatient.study.registration.set_representative_slices;
        aPatient.study.registration.plot_overlays;
        aPatient.study.registration.report;
        
     %Add model
     aPatient.study.add_model;
        aPatient.study.model.fit;
        aPatient.study.model.plot_residual;
        aPatient.study.model.report;
        
            %Add sequence
            
        
        
        
    