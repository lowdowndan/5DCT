%% Workflow script

new_patient;

    % Add patient
    aPatient = patient(id);
    
    % Add study
    aPatient.add_study;
    aPatient.study.synchronize;
    
    % Add registration
    aPatient.study.add_registration;
        aPatient.study.registration.register;
        aPatient.study.registration.slice;
        aPatient.study.registration.set_representative_slices;
        
     %Add model
     aPatient.study.add_model;
        aPatient.study.model.fit;
        aPatient.study.model.plot_residual;
        
        
    