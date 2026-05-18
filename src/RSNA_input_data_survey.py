# coding utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import concurrent.futures
import multiprocessing
import SimpleITK as sitk
from typing import List, Set, Dict, Any, Tuple
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
def _print_and_log(
    msg: str,
    logger: logging.Logger,
    level: int = logging.INFO
) -> None:
    """
    Outputs a message to the standard console and records it via the logger.

    This utility unifies console feedback and persistent logging into a single
    call, supporting dynamic severity levels (e.g., INFO, WARNING, CRITICAL).

    Args:
        msg: The string message to be displayed and logged.
        logger: The logging.Logger instance to handle the record.
        level: The logging severity level. Defaults to logging.INFO.
    """
    print(msg)
    logger.log(level, msg)


def _get_labeled_files(csv_path: str) -> set:
    """
    Parses a CSV file to extract ground truth coordinates for quick validation.

    This function reads the specified CSV and isolates the unique identifiers
    for studies, series, and instances. By converting these into a set of
    integer tuples, it enables O(1) time complexity for membership checks
    during DICOM analysis.

    Args:
        csv_path: System path to the CSV file containing labels.
            Expected columns: 'study_id', 'series_id', 'instance_number'.

    Returns:
        A set of tuples, where each tuple is (study_id, series_id, instance_number).
    """
    df = pd.read_csv(csv_path)[['study_id', 'series_id', 'instance_number']]
    return set(df.astype(int).itertuples(index=False, name=None))


def _analyze_single_series(
    study_path: Path,
    series: Path,
    reader: sitk.ImageFileReader,
    labeled_set: Set[Tuple[int, int, int]],
    results: Dict[str, Any]
) -> None:

    """
    Analyzes all DICOM files within a specific series and updates the study results.

    This function performs the second level of nesting in the analysis pipeline.
    It counts the slices (depth), invokes the file-level processor for metadata
    extraction, and calculates the consistency of image formats (dimensions)
    within this specific series.

    Args:
        study_path: Path to the parent study directory.
        series: Path to the specific series directory to analyze.
        reader: SimpleITK ImageFileReader instance for metadata extraction.
        labeled_set: Set of (study, series, instance) tuples for label validation.
        results: Local or global results dictionary to update with findings.
        logger: Logger instance for reporting inconsistencies (can be None).

    Note:
        This function updates the 'results' dictionary in-place, specifically
        modifying 'depths' and 'consistency' keys.
    """

    # Identify all DICOM files in the series
    dcm_files = list(series.glob('*.dcm'))
    results["depths"].append(len(dcm_files))

    unique_fmts, unique_spcs = set(), set()

    # Delegates individual file processing
    _process_files(
        dcm_files,
        reader,
        unique_fmts,
        unique_spcs,
        study_path,
        series,
        labeled_set,
        results
    )

    # Calculate and record how many different formats exist in the series.
    # A perfectly consistent series should hve exactly 1 unique format.
    nb_f = len(unique_fmts)
    results["consistency"][nb_f] = results["consistency"].get(nb_f, 0) + 1

    if nb_f > 1:
        results["logs"].append((
            "warning",
            f"Inconsistency detected in {series}: {nb_f} different formats found."
        ))


def _analyze_single_study(
    study_path: Path,
    labeled_set: Set[Tuple[int, int, int]]
) -> Dict[str, Any]:

    """
    Analyzes one study and returns a standalone results dictionary.
    Isolated for future parallel execution.
    """
    # Local results for this worker
    results = {
        "depths": [],
        "formats": {},
        "spacings": {},
        "consistency": {},
        "min": float('inf'),
        "max": float('-inf'),
        "logs": []
    }
    reader = sitk.ImageFileReader()

    try:
        for series in [s for s in study_path.iterdir() if s.is_dir()]:
            # Ensure _analyze_single_series handles logs by adding to results["logs"]
            _analyze_single_series(study_path, series, reader, labeled_set, results)

        # Success log
        info_msg = f"Successfully processed study: {study_path.name}"
        results["logs"].append(("info", info_msg))

    except Exception as e:
        # Catch and store error without crashing the worker process
        error_msg = f"Error processing study {study_path.name}: {e}"
        results["logs"].append(("error", f"{error_msg}"))

    return results


def _analyze_dataset_dicom_files(
    studies_list: List[Path],
    labeled_set: Set[Tuple[int, int, int]],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Executes a comprehensive scan of DICOM studies to evaluate dataset integrity.

    This function iterates through study and series directories to collect metadata
    and pixel statistics. It identifies format inconsistencies within series and
    cross-references them with ground truth labels to flag critical data issues.

    Args:
        studies_list: A list of Path objects pointing to individual study directories.
        labeled_set: A set of (study_id, series_id, instance_number) tuples used
            for O(1) lookup of ground truth references.
        logger: A logging.Logger instance for status updates and error reporting.

    Returns:
        A dictionary containing global statistics:
            - "depths" (list): Number of slices per series.
            - "formats" (dict): Distribution of image dimensions (PixelSize).
            - "spacings" (dict): Distribution of pixel spacings.
            - "consistency" (dict): Count of unique formats found per series.
            - "min"/"max" (float): Global pixel intensity boundaries.
    """
    global_results = {
        "depths": [],
        "formats": {},
        "spacings": {},
        "consistency": {},
        "min": float('inf'),
        "max": float('-inf')
    }

    # Define the number of workers (leaving one core free for system stability)
    max_workers = multiprocessing.cpu_count() - 1

    # Using ProcessPoolExecutor to bypass the "Global Interpreter Lock" (GIL)
    # and utilize multiple CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:

        # Dispatching tasks (Map)
        # We submit each study analysis as an independent process
        future_to_study = {
            executor.submit(_analyze_single_study, study_path, labeled_set): study_path
            for study_path in studies_list
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_study),
            total=len(studies_list),
            desc="Parallel Processing",
            unit="study"
        ):

            study_path = future_to_study[future]
            try:
                # Retrieve the result dictionary from the worker process
                study_res = future.result()

                # Dump the logs gathered by the worker
                for level, message in study_res.get("logs", []):
                    if level == "critical":
                        logger.critical(message, exc_info=True)
                    elif level == "error":
                        logger.error(message, exc_info=True)
                    elif level == "warning":
                        logger.warning(message)
                    else:
                        logger.info(message)

                # AGGREGATION LOGIC (The "Reduce" step)
                global_results["depths"].extend(study_res["depths"])
                global_results["min"] = min(global_results["min"], study_res["min"])
                global_results["max"] = max(global_results["max"], study_res["max"])

                # Merge frequency distribution dictionaries
                for key in ["formats", "spacings", "consistency"]:
                    for k, v in study_res[key].items():
                        global_results[key][k] = global_results[key].get(k, 0) + v

            except Exception as e:
                # Handle potential crashes within a specific worker process
                logger.error(f"Study {study_path.name} generated an exception: {e}")
                # Optional: Depending on Fail-Fast preference, you could 'raise' here
                # or continue to process remaining studies.

    return global_results


def _format_inconsistency_report(
    study: Path,
    series: Path,
    dcm_file: Path,
    pixel_size: Tuple[int, ...],
    spacing: Tuple[float, ...],
    unique_spacings: Set[Tuple[float, ...]],
    unique_formats: Set[Tuple[int, ...]],
    labeled_files_set: Set[Tuple[int, int, int]]
) -> str:
    """
    Constructs a detailed string report of metadata discrepancies.

    Returns:
        A multi-line string containing the formatted report.
    """
    study_id_val = int(study.stem)
    series_id_val = int(series.stem)
    instance_val = int(dcm_file.stem)
    fov_x = pixel_size[0] * spacing[0]
    fov_y = pixel_size[1] * spacing[1]

    lines = [
        "--- Inconsistency Detected ---",
        f"Study: {study_id_val} | Series: {series_id_val} | File: {instance_val}"
    ]

    if (study_id_val, series_id_val, instance_val) in labeled_files_set:
        lines.append("!!! CRITICAL: This inconsistent file is a LABEL reference !!!")

    lines.append(f"Dimensions: {pixel_size} pixels")
    lines.append(f"Pixel Spacing: {spacing[0]:.1f} x {spacing[1]:.1f} mm²")
    lines.append(f"Real FOV: {fov_x:.1f} x {fov_y:.1f} mm²")
    lines.append(f"History - Spacings: {unique_spacings} | Formats: {unique_formats}\n")

    return "\n".join(lines)


def _process_files(
    files: List[Path],
    reader: sitk.ImageFileReader,
    unique_fmts: Set[Tuple[int, ...]],
    unique_spcs: Set[Tuple[float, ...]],
    study: Path,
    series: Path,
    labeled_set: Set[Tuple[int, int, int]],
    results: Dict[str, Any]
) -> None:
    """
    Analyzes individual DICOM files within a series to update dataset statistics.

    This internal routine extracts geometric metadata (spacing, size) and pixel
    intensity ranges for each file. It triggers inconsistency logging if a file's
    dimensions deviate from the previously encountered formats in the same series.

    Args:
        files: List of DICOM file paths in the current series.
        reader: Reusable SimpleITK ImageFileReader instance.
        unique_fmts: Tracking set for image dimensions (width, height) in the series.
        unique_spcs: Tracking set for pixel spacings in the series.
        study: Path to the parent study directory.
        series: Path to the current series directory.
        labeled_set: Set of (study, series, instance) IDs for label cross-referencing.
        results: Local results dictionary to be updated with formats and pixel ranges.
    """
    for dcm in files:
        try:
            reader.SetFileName(str(dcm))
            reader.ReadImageInformation()

            spacing, size = reader.GetSpacing(), reader.GetSize()

            # Detect Inconsistency
            if len(unique_fmts) > 0 and size not in unique_fmts:
                # Generate a detailed report string instead of direct logging
                report = _format_inconsistency_report(
                    study,
                    series,
                    dcm,
                    size,
                    spacing,
                    unique_spcs,
                    unique_fmts,
                    labeled_set
                )
                results["logs"].append(("warning", report))

            unique_fmts.add(size)
            unique_spcs.add(spacing)
            results["formats"][size] = results["formats"].get(size, 0) + 1
            results["spacings"][spacing] = results["spacings"].get(spacing, 0) + 1

            # Pixel range
            pixels = sitk.GetArrayFromImage(reader.Execute())
            results["min"] = min(results["min"], np.min(pixels))
            results["max"] = max(results["max"], np.max(pixels))

        except Exception:
            # Silently skip corrupted or unreadable DICOM files
            continue


def _report_statistics(
    data: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Summarizes dataset analysis and provides storage recommendations.

    This function processes the aggregated results to display distribution
    statistics for file formats and pixel intensities. It includes a logic
    to suggest the most efficient NumPy/TensorFlow data type based on the
    global dynamic range of the pixel values.

    Args:
        data: Dictionary containing aggregated stats ('formats', 'min', 'max', 'consistency').
        logger: Logger instance for standardized console and file output.
    """

    # 1. Display file format distribution
    _print_and_log(
        "\n--- Statistics for File Format Distribution ---\n",
        logger,
        level=logging.INFO
    )

    for fmt, count in data["formats"].items():
        _print_and_log(
            f"\tFormat {fmt}: {count} {'file' if count <= 1 else 'files'}",
            logger,
            level=logging.INFO
        )

    # 2. Display pixel intensity range and storage advice
    _print_and_log(
        "\n--- Series Consistency Summary ---",
        logger,
        level=logging.INFO
    )

    _print_and_log(
        f"Global Min: {data['min']} | Global Max: {data['max']}",
        logger,
        level=logging.INFO
    )

    # Recommended storage format
    suggestion = "int16"
    if data["min"] >= 0:
        suggestion = "uint8" if data["max"] <= 255 else "uint16"
    elif -128 <= data["min"] and data["max"] <= 127:
        suggestion = "int8"

    _print_and_log(
        f"Recommended Storage Format: {suggestion}",
        logger,
        level=logging.INFO
    )

    # 3. Display series consistency table using Pandas for formatting
    df_stats = pd.DataFrame(
        list(data["consistency"].items()),
        columns=['Unique Formats per Series', 'Number of Series']
    )

    formatted_table = df_stats.sort_values('Unique Formats per Series').to_string(index=False)
    _print_and_log(
        formatted_table,
        logger,
        level=logging.INFO
    )


def _plot_distribution(
    depth_list: list,
    logger: logging.Logger
) -> None:
    """
    Visualizes the dataset depth distribution with a dual-axis chart.

    This function generates a 20-bin histogram showing the frequency of series
    depths alongside a cumulative step-line plot. This dual representation
    helps identify both common slice counts and the overall sample size
    coverage across the dataset.

    Args:
        depth_list: A list of integers representing the number of slices per series.
        logger: Logger instance (used for consistency in the pipeline architecture).
    """
    # Initialize the figure with a dual-axis layout
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 1. Plot Histogram (Main axis)
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
    ax1.set_ylabel("Frequency", color='blue')

    # 2. Plot Cumulative frequency (Secondary Y-axis)
    ax2 = ax1.twinx()
    cumulative_counts = np.cumsum(counts)
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
    ax2.plot(bin_centers, cumulative_counts, color='red', marker='o', label='Cumulative Count')
    ax2.set_ylabel("Total Sample Size (Cumulative)", color='red')

    # Final styling and display
    plt.title(f"Distribution of Series Depth (n={len(depth_list)})")
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Merge legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.tight_layout()
    plt.show()


def main():
    """
    Orchestrates the DICOM dataset inspection pipeline.

    The workflow consists of four main stages:
    1. Setup: Resolves directory paths, filters study folders, and loads
       the ground truth labels for inconsistency cross-referencing.
    2. Execution: Performs a deep scan of the DICOM studies to collect
       metadata (dimensions, spacing) and pixel value ranges.
    3. Reporting: Logs a statistical synthesis of the dataset, including
       consistency checks and storage format recommendations.
    4. Visualization: Generates a combined histogram and cumulative
       distribution plot of series depths.
    """

    # 1. Setup
    dicom_dir = Path(DICOM_STUDIES_DIR).resolve()
    log_dir = Path(config["paths"].get("inspection", "logs")) / "logs"
    studies = [s for s in dicom_dir.iterdir() if s.is_dir()]
    labeled_set = _get_labeled_files(CSV_LABEL_COORDINATES)

    with setup_logger(process_name="train", log_dir=log_dir) as logger:
        _print_and_log("STARTING ANALYSIS", logger, level=logging.INFO)

        # 2. Execution
        data = _analyze_dataset_dicom_files(studies, labeled_set, logger)

        # 3. Reporting
        _report_statistics(data, logger)

        # 4. Visualization
        _plot_distribution(data["depths"], logger)


# Entry point
if __name__ == '__main__':
    main()
