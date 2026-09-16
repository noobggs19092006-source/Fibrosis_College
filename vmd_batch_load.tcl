# Set the directory containing the 1630 PDBs
set pdb_dir "/media/aryan-k-k/2TB_KK/Fibrosis_College/pdb_structures"

# Get a list of all .pdb files in that directory
set files [glob -directory $pdb_dir *.pdb]

# Counter to keep track
set count 0

# Loop through and load each file
foreach f $files {
    mol new $f waitfor all
    incr count
}
puts "Successfully loaded $count molecules!"
