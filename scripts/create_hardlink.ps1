# Define the absolute path to the configuration directory
$ConfigDir = "C:\Users\Olivier\Desktop\Projet_Kaggle\RSNA_Items_Detection_Classification\src\config"

# 1. Forcefully remove the existing configuration file or link to avoid conflicts
Remove-Item -Path "$ConfigDir\lumbar_spine_config.yaml" -Force

# 2. Create a Hard Link pointing to the Windows-specific configuration file.
#    This alternative does not require local Administrator privileges,
#    making it seamlessly usable by any standard user profile
New-Item -Path "$ConfigDir\lumbar_spine_config.yaml" -ItemType HardLink -Value "$ConfigDir\lumbar_spine_config_windows.yaml"