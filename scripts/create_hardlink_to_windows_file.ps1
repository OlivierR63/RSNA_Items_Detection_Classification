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

# 1. Determine the path to the configuration directory.
#    Using $PSScriptRoot allows the script to remain portable across different machines.
#    Adjust the relative path structure below depending on where this .ps1 file is stored.
if ($PSScriptRoot) {
    # Example structure: if script is in "[project_root]/scripts/" or similar, adjust the parent levels
    $ConfigDir = Join-Path (Get-Item $PSScriptRoot).Parent.FullName "src\config"
} else {
    # Fallback to the absolute path if executed line-by-line in a terminal without a file context
    $ConfigDir = "C:\Users\Olivier\Desktop\Projet_Kaggle\RSNA_Items_Detection_Classification\src\config"
}

# Ensure the calculated directory actually exists before proceeding
if (-not (Test-Path -Path $ConfigDir)) {
    Write-Error "Target configuration directory not found: $ConfigDir"
    Exit 1
}

$TargetLink = Join-Path $ConfigDir "lumbar_spine_config.yaml"
$SourceFile = Join-Path $ConfigDir "lumbar_spine_config_windows.yaml"

# 2. Forcefully remove the existing configuration file or link to avoid conflicts.
#    -ErrorAction SilentlyContinue prevents crashing if the file does not exist yet.
Remove-Item -Path $TargetLink -Force -ErrorAction SilentlyContinue

# 3. Create a Hard Link pointing to the Windows-specific configuration file.
#    This alternative does not require local Administrator privileges,
#    making it seamlessly usable by any standard user profile.
New-Item -Path $TargetLink -ItemType HardLink -Value $SourceFile | Out-Null

Write-Host "Successfully linked active configuration to Windows setup profile." -ForegroundColor Green