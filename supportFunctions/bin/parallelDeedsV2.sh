#!/bin/bash
# parallelDeeds

# Registers CT scans to reference scan using deedsMIND algorithm.
# Excecutes numJobs registrations simulatenously 

# See if user input all parameters
if [ "$#" -ne 4 ]; then
echo "Usage: parallelDeedsV2 inputDirectory outputDirectory referenceScanNumber numJobs"
exit
fi

if [ ! -d $2 ]; then
# Make output directory
mkdir $2
fi
# Get list of files in inputDirectory
shopt -s nullglob
declare -a scanFilenames
scanFilenames=($1/*.nii)
shopt -u nullglob

# Execute registrations
printf -v refScan '%02d.nii' $3

refScan=$1/$refScan
# Comment this out to limit the number of processes by CPU usage rather than number 
# of jobs
ls $1/*nii | parallel --progress -j$4 deedsMIND $refScan {} $2/{/.} 2.0 128.0

# Uncomment to use the 4th input argument as a maximum percentage of CPU resources
# to use.  The script will only launch another registration if cpu usage is under
# this percentage.
# e.g.: parallelDeeds inputDir outputDir refScanNumber 80
# would only launch additional registrations if CPU usage is less than 80%
#ls $1/*nii | parallel --progress --load $4% deedsMIND $refScan {} $2/{/.} 2.0 128.0



# Remove the "_deformed" appended to the output image from deedsMIND and call resizeFlow
ls $2/*deformed.nii | sed 's/.\{13\}$//' | parallel --progress -j$4 resizeFlow {.}  
  
