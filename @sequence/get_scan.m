%%

function aScan = get_scan(aSequence, phaseDesc)
% Verify that the reconstruction points have been set
assert(isequal(numel(aSequence.reconstructionPoints), numel(aSequence.inhaleAmplitudes) + numel(aSequence.exhaleAmplitudes)),'Reconstruction points have not been set.  Run set_reconstruction_points method.');

descList = {aSequence.reconstructionPoints.description};
descMatch = contains(descList,phaseDesc,'IgnoreCase',1);
descMatch = find(descMatch,1,'first');

aScan = load(aSequence.scans{descMatch});
scanName = fieldanmes(aScan);
aScan = aScan.(scanName{1});

end
