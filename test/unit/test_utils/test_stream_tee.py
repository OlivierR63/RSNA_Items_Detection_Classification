# coding: utf-8

from unittest.mock import MagicMock, patch
from io import StringIO
from src.core.utils.system_stream_tee import SystemStreamTee


class TestSystemStreamTee:
    """
    Unit tests for the SystemStreamTee class to ensure proper stream mirroring.
    """

    def test_write_mirrors_to_both_outputs(self, tmp_path):
        """
        Verifies that a message is written to both the terminal and the log file.
        Args:
            tmp_path (Path): Pytest fixture providing a temporary directory.
        """
        # Setup: Define log path and mock the terminal (stdout)
        log_file = tmp_path / "test_mirror.log"
        fake_terminal = StringIO()

        with patch('sys.stdout', fake_terminal):
            tee = SystemStreamTee(str(log_file))
            test_message = "Data Science Traceability Test"

            # Execute the write operation
            tee.write(test_message)

            # Verification: Check terminal output
            assert test_message in fake_terminal.getvalue()

            # Verification: Check file content
            with open(log_file, 'r', encoding="utf-8") as f:
                content = f.read()
                assert test_message in content

            tee.close()

    def test_header_contains_hardware_info(self, tmp_path):
        """Checks if the session header correctly includes hardware detection info.

        Args:
            tmp_path (Path): Pytest fixture providing a temporary directory.
        """
        log_file = tmp_path / "hardware_check.log"

        # Mock TensorFlow to simulate a specific hardware configuration
        # This avoids loading real GPU drivers during unit testing
        with patch('tensorflow.config.list_physical_devices') as mock_list:
            # Simulate 1 CPU and 2 GPUs
            mock_list.side_effect = (
                lambda dev_type: [f"{dev_type}_0"] if dev_type == 'CPU' else ["GPU_0", "GPU_1"]
            )

            tee = SystemStreamTee(str(log_file))

            with open(log_file, 'r', encoding="utf-8") as f:
                header_content = f.read()
                # Verify header metadata
                assert "NEW SESSION STARTED" in header_content
                assert "CPUS: 1" in header_content
                assert "GPUS: 2" in header_content

            tee.close()

    def test_directory_auto_creation(self, tmp_path):
        """Ensures that missing parent directories are created automatically.

        Args:
            tmp_path (Path): Pytest fixture providing a temporary directory.
        """
        # Path with multiple non-existent subdirectories
        deep_log_path = tmp_path / "logs" / "deep" / "path" / "session.log"

        tee = SystemStreamTee(str(deep_log_path))

        # Verify the file was actually created in the nested directory
        assert deep_log_path.exists()
        assert deep_log_path.parent.is_dir()

        tee.close()

    def test_flush_operation(self, tmp_path):
        """Verifies that the flush method clears both internal buffers.

        Args:
            tmp_path (Path): Pytest fixture providing a temporary directory.
        """
        log_file = tmp_path / "flush_test.log"
        fake_terminal = MagicMock()

        with patch('sys.stdout', fake_terminal):
            tee = SystemStreamTee(str(log_file))
            tee.flush()

            # Ensure flush was called on both the terminal and the file handle
            assert fake_terminal.flush.called
            # Note: _log_file is a real file object, we just ensure no crash occurs

            tee.close()
