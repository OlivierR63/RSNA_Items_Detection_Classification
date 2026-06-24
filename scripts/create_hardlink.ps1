<#
.SYNOPSIS
    Configures the environment's active configuration file using a hard link.

.DESCRIPTION
    This script automates the configuration switching process for a standard local Windows 
    development environment. It ensures that the generic configuration file name 
    ('lumbar_spine_config.yaml') used by the application core points directly to the 
    Windows-specific setup file ('lumbar_spine_config_windows.yaml').
    
    By utilizing a Hard Link instead of a Symbolic Link (SymLink), this script can be executed 
    seamlessly by standard user accounts without requiring elevated local Administrator privileges.

.NOTES
    Context: RSNA Lumbar Spine Degenerative Classification project.
#>

# Define the absolute path to the configuration directory
$ConfigDir = "C:\Users\Olivier\Desktop\Projet_Kaggle\RSNA_Items_Detection_Classification\src\config"

# 1. Forcefully remove the existing configuration file or link to avoid conflicts
Remove-Item -Path "$ConfigDir\lumbar_spine_config.yaml" -Force

# 2. Create a Hard Link pointing to the Windows-specific configuration file.
#    This alternative does not require local Administrator privileges,
#    making it seamlessly usable by any standard user profile
New-Item -Path "$ConfigDir\lumbar_spine_config.yaml" -ItemType HardLink -Value "$ConfigDir\lumbar_spine_config_windows.yaml"