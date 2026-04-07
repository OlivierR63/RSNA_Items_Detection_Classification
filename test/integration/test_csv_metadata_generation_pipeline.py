# coding: utf-8

import pandas as pd
import numpy as np
from src.projects.lumbar_spine.csv_metadata_handler import CSVMetadataHandler


def test_merge_metadata_integration(setup_csv_files, mock_logger, mock_config, caplog):
    """
    Test the complete metadata merge pipeline.
    Ensures that the final DataFrame contains all necessary features
    and represents the correct intersection of input files.
    """
    files = setup_csv_files

    # 1. Initialize the handler
    # Note: initialization triggers _load_and_cleanse_data automatically
    handler = CSVMetadataHandler(
        dicom_studies_dir="",
        series_description=files["series_description"],
        label_coordinates=files["label_coordinates"],
        label_enriched=files["label_enriched"],
        train=files["train"],
        config=mock_config,
        logger=mock_logger
    )

    # 2. Execute the merge process
    merged_df = handler._merge_metadata()

    # --- ASSERTIONS ---

    # A. Structural Integrity
    assert isinstance(merged_df, pd.DataFrame)
    assert not merged_df.empty, "The merged DataFrame should not be empty."

    # B. Column Presence Verification
    # These columns are the result of merging train, coordinates, and series descriptions
    required_columns = [
        "study_id", "condition_level", "severity",
        "series_id", "instance_number", "x", "y"
    ]
    for col in required_columns:
        assert col in merged_df.columns, f"Merged DataFrame is missing column: {col}"

    # C. Data Consistency Logic
    # Verify that study_id is normalized as integer
    assert pd.api.types.is_integer_dtype(merged_df["study_id"])

    # Verify that severity text is cleaned (lowercase and stripped)
    sample_severity = merged_df["severity"].iloc[0]
    assert sample_severity == sample_severity.lower().strip()

    # D. Verification of the Merge Intersection (The "Shape" Check)
    # We manually simulate the expected intersection to compare counts.
    # We expect only rows that exist in BOTH melted train data AND coordinates.
    melted_train = handler._train_df.melt(
        id_vars="study_id",
        var_name="condition_level",
        value_name="severity"
    ).dropna()

    # Force study_id to int32 to match handler._label_coords_df
    melted_train["study_id"] = melted_train["study_id"].astype(np.int32)

    # Note: condition_level in label_coords is already formatted by the handler
    expected_count = len(pd.merge(
        melted_train,
        handler._label_coords_df,
        on=["study_id", "condition_level"],
        how="inner"
    ))

    assert merged_df.shape[0] == expected_count, (
        f"Row count mismatch. Expected {expected_count} rows (inner join), "
        f"but got {merged_df.shape[0]}."
    )

    # E. Successful Termination Check
    # Confirming the logger reached the final success message
    expected_message = "CSVMetadataHandler._merge_metadata completed successfully"
    failure_message = f"The success message '{expected_message}' was not found in the logs."
    assert any(expected_message in record.message for record in caplog.records), failure_message
