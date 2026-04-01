# coding utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from src.config.config_loader import ConfigLoader
from src.core.utils.logger import setup_logger

# 2. Configuration Loading
config_loader = ConfigLoader("src/config/lumbar_spine_config.yaml")
config: dict = config_loader.get()

# 3.  Constant assignments (After functions are defined)
MAX_RECORDS = config['data_specs'].get('max_records_per_frame', None)
if MAX_RECORDS is None:
    error_msg = (
        "Fatal error: the setting variables 'data_specs -> max_records_per_frame' "
        "is required but was not found. Please check your YAML file structure."
    )
    raise ValueError(error_msg)

DICOM_STUDIES_DIR = config['paths']['dicom_studies']
CSV_LABEL_COORDINATES = config['paths']['csv']['label_coordinates']


# Function for printing the information messages in the console and
# saving them in parallel in the log file.
def print_and_log_info(
    msg: str,
    logger: logging.Logger
) -> None:
    print(msg)
    logger.info(msg)


def print_and_log_critical(
    msg: str,
    logger: logging.Logger
) -> None:
    print(msg)
    logger.critical(msg)


# Build and display statistics on series depth.
def main():
    """
    Calculates the maximum number of slices per series.
    Verifies the format consistency.
    Verifies if the pixel spacing remains constant on every files or not.
    """

    dicom_studies_dir = Path(DICOM_STUDIES_DIR).resolve()
    nb_dicom_files = len(list(dicom_studies_dir.rglob('*.dcm')))
    studies_dirs_list = [study for study in dicom_studies_dir.iterdir() if study.is_dir()]

    depth_list = []

    log_dir = config["paths"].get("inspection", "logs")  # use "logs" as default if not in config.
    log_dir += "/logs"

    # This file stores the coordinates of the observed pathologies (condition)
    csv_label_coordinates_df = pd.read_csv(CSV_LABEL_COORDINATES)
    csv_coordinates_files_df = csv_label_coordinates_df[
        ['study_id', 'series_id', 'instance_number']
    ]

    # Convert labels to a set of tuples for O(1) lookup
    # Format: {(study_id, series_id, instance_number), ...}
    labeled_files_set = set(
        csv_coordinates_files_df.astype(int).itertuples(index=False, name=None)
    )

    with setup_logger("train", log_dir=log_dir, config=config) as logger:

        print_and_log_info("\n" + "="*40, logger)
        print_and_log_info("STARTING ANALYSIS OF STUDIES FOLDER", logger)
        print_and_log_info("="*40, logger)

        # Dictionary to store: {nb_of_unique_formats: count_of_series}
        format_consistency_stats = {}

        reader = sitk.ImageFileReader()

        overall_format_dict = {}
        overall_spacing_dict = {}

        for study in tqdm(studies_dirs_list, desc="Processing Studies", unit="study"):
            # 1. Calculate depth for each series in study
            study_series = [s for s in study.iterdir() if s.is_dir()]

            for series in study_series:
                dcm_files = list(series.glob('*.dcm'))
                depth_list.append(len(dcm_files))

                # 2. Check format consistency within the series
                unique_formats = set()
                unique_spacings = set()
                for dcm_file in dcm_files:
                    try:
                        reader.SetFileName(str(dcm_file))
                        reader.ReadImageInformation()
                        size_before = len(unique_formats)

                        # Retrieve the Pixel Spacing (in mm/pixel)
                        # Return (spacing_x, spacing_y, spacing_z)
                        spacing = reader.GetSpacing()
                        unique_spacings.add(spacing)
                        overall_spacing_dict[spacing] = overall_spacing_dict.get(spacing, 0) + 1

                        # Retrieve the dimensions (in pixels)
                        pixel_size = reader.GetSize()
                        unique_formats.add(pixel_size)
                        size_after = len(unique_formats)
                        overall_format_dict[pixel_size] = overall_format_dict.get(pixel_size, 0) + 1

                        # Calculate the actual field of view (FoV, in mm)
                        fov_x = pixel_size[0] * spacing[0]
                        fov_y = pixel_size[1] * spacing[1]

                        if (size_before > 0) and (size_before < size_after):
                            study_id_val = int(study.stem)
                            series_id_val = int(series.stem)
                            instance_val = int(dcm_file.stem)

                            print_and_log_info("--- Inconsistency Detected ---", logger)
                            print_and_log_info(
                                (
                                    f"Study: {study_id_val} | "
                                    f"Series: {series_id_val} | "
                                    f"File: {instance_val}"
                                ),
                                logger
                            )

                            # Instant check if this specific inconsistent file
                            # is a ground truth label
                            if (study_id_val, series_id_val, instance_val) in labeled_files_set:
                                print_and_log_critical(
                                    "This inconsistent file is a LABEL reference!",
                                    logger
                                )

                            print_and_log_info(
                                f"Dimensions: {pixel_size} pixels",
                                logger
                            )
                            print_and_log_info(
                                f"Pixel Spacing: {spacing[0]:.1f} x {spacing[1]:.1f} mm",
                                logger
                            )
                            print_and_log_info(
                                f"Real FOV: {fov_x:.1f} x {fov_y:.1f} mm",
                                logger
                            )
                            print_and_log_info(
                                f"Pixel Spacing before and after (if different): {unique_spacings}",
                                logger
                            )
                            print_and_log_info(
                                f"Format before and after :{unique_formats}\n\n",
                                logger
                            )

                    except Exception:
                        continue  # Skip corrupted headers

                # Count how many unique formats this specific series has
                nb_formats = len(unique_formats)
                format_consistency_stats[nb_formats] = (
                    format_consistency_stats.get(nb_formats, 0) + 1
                )

        # Synthesis of the analysis
        print_and_log_info("\n--- Statistics for file format Distribution ---\n", logger)
        img_count = 0
        for img_format, nb_files in overall_format_dict.items():
            files = "file" if nb_files <= 1 else "files"
            print_and_log_info(f"\tFormat {img_format}: {nb_files} {files}", logger)
            img_count += nb_files

        print_and_log_info(f"\n\tThe total count of dicom files is {img_count}", logger)

        print_and_log_info("\n\n--- Statistics for Pixel Spacing Distribution ---\n", logger)
        img_count = 0
        for spacing, nb_files in overall_spacing_dict.items():
            files = "file" if nb_files <= 1 else "files"
            print_and_log_info(f"\tPixel Spacing {spacing}: {nb_files} {files}", logger)
            img_count += nb_files

        print_and_log_info(f"\n\tThe total count of dicom files is {img_count}", logger)

        # --- Display Summary Table ---
        print_and_log_info("\n\n" + "="*40, logger)
        print_and_log_info("SERIES CONSISTENCY SUMMARY", logger)
        print_and_log_info("="*40, logger)

        # Create a DataFrame for clean display
        df_stats = pd.DataFrame(list(format_consistency_stats.items()),
                                columns=['Unique Formats per Series', 'Number of Series'])
        df_stats = df_stats.sort_values(by='Unique Formats per Series')

        print_and_log_info(df_stats.to_string(index=False), logger)
        print_and_log_info("="*40, logger)

        # Plotting section:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 1. Plot the Histogram (Primary Axis)
        counts, bins_edges, _ = ax1.hist(
            depth_list,
            bins=20,
            range=(0, 200),
            color='skyblue',
            edgecolor='black',
            alpha=0.7,
            label='Frequency'
        )
        ax1.set_xlabel("Depth (Number of Slices)")
        ax1.set_ylabel("Frequency (Individual)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Display the values in the console
        print_and_log_info("\n\n--- Statistics for Series Depth Distribution ---\n", logger)
        for idx in range(len(counts)):
            print_and_log_info(
                f"\tBin {idx+1} [{bins_edges[idx]:.0f} - {bins_edges[idx+1]:.0f}]: "
                f"{int(counts[idx])} series",
                logger
            )

        print_and_log_info(f"\n\tThe total count of dicom files is {nb_dicom_files}", logger)

        # 2. Calculate Cumulative Frequency
        # np.cumsum adds elements progressively: [c1, c1+c2, c1+c2+c3, ...]
        cumulative_counts = np.cumsum(counts)

        # We use the center of each bin for the scatter points
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

        # 3. Create a secondary Y-axis for the cumulative plot
        ax2 = ax1.twinx()

        # Plotting the scatter points connected by solid lines
        ax2.plot(bin_centers, cumulative_counts, color='red', marker='o',
                 linestyle='-', linewidth=2, label='Cumulative Count')

        ax2.set_ylabel("Total Sample Size (Cumulative)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Add a grid for better readability under Visual Studio
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # 4. Final adjustments
        plt.title(f"Distribution of Series Depth (n={len(depth_list)})")

        # Merging legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc='center left',
            bbox_to_anchor=(1.15, 0.5)
         )

        # Adjust the layout to prevent the legend from being cut off
        plt.tight_layout()

        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.show()


# Entry point
if __name__ == '__main__':
    main()
