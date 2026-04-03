# coding: utf-8

import os
import sys
from pathlib import Path
from shutil import rmtree
from git import Repo

# Configuration flags
UPDATE_REPO = True

# Repository settings
repo_url = "https://github.com/OlivierR63/RSNA_Items_Detection_Classification"
repo_path = Path("/kaggle/working/RSNA_Items_Detection_Classification")


def setup_kaggle_environment():
    """
    Handles repository cloning and environment setup on Kaggle.
    """
    # 1. GitHub Repository Management
    if UPDATE_REPO:
        if repo_path.exists() and repo_path.is_dir():
            # Remove existing directory to ensure a clean slate
            rmtree(repo_path)
            print(f"Directory {repo_path} removed successfully.")

        print("_" * 50)
        print("         Cloning GitHub repository")
        print("_" * 50)

        # Ensure we are in the working directory
        os.chdir("/kaggle/working")

        # Clone using GitPython (Pure Python)
        Repo.clone_from(repo_url, repo_path)
        print("         Cloning completed")

    # 2. Path definitions for Symbolic Link
    target_physical_file = repo_path / "src/config/lumbar_spine_config_kaggle.yaml"
    link_to_create = repo_path / "src/config/lumbar_spine_config.yaml"

    # 3. Create the Symbolic Link
    if link_to_create.exists() or link_to_create.is_symlink():
        link_to_create.unlink()

    # Link the generic config name to the Kaggle-specific file
    link_to_create.symlink_to(target_physical_file)
    print(f"Symbolic link created: {link_to_create.name} -> {target_physical_file.name}\n")

    # 4. Launch the application
    # Point to the root of the project (the parent of 'src')
    project_root = repo_path
    os.chdir(project_root)  # Go to /kaggle/working/RSNA_Items_Detection_Classification

    # Add the project root to sys.path so 'src' can be found as a module
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Path to the main script
    main_script = project_root / "src/RSNA_2024_Lumbar_Spine_Degenerative_Classification.py"

    with open(main_script, "r", encoding="utf-8") as f:
        # We must provide the current globals() to exec so imports work correctly
        # and keep the context of the notebook
        ctx = globals().copy()
        ctx.update({"__name__": "__main__", "__file__": str(main_script)})
        exec(f.read(), ctx)


if __name__ == "__main__":
    setup_kaggle_environment()
