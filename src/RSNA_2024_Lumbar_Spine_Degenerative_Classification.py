import signal
from src.core.utils.logger import setup_logger, get_current_logger, log_method
from src.core.utils.clean_logs import clean_old_logs
from src.projects.lumbar_spine.train import train_model
from src.config.config_loader import ConfigLoader
import sys
import logging


def handle_interrupt(signum, frame):
    """Handles interrupt signals (Ctrl+C) to ensure proper log closure."""
    try: 
        logger = get_current_logger()
        logger.info("\nInterruption detected (Ctrl+C). Exiting gracefully...")
    except RuntimeError:
        print("\nInterruption detected (Ctrl+C). Exiting gracefully...")    
    
    sys.exit(0)


def main():
    """
    Main function to load configuration, set up the TensorFlow dataset pipeline,
    load and compile the 3D model, and start the training process.
    """
    # Setup signal handler for graceful interruption
    signal.signal(signal.SIGINT, handle_interrupt)

    # 1. Load the Configuration
    config_loader = ConfigLoader("src/config/lumbar_spine_config.yaml")
    config:dict = config_loader.get()
    

    # 2. Initialize logger with process-specific context
    log_dir = config.get("output_dir", "logs")   #use "logs" as default if not in config.
    log_dir += "/logs"

    with setup_logger("train", log_dir=log_dir, use_json=True) as logger:
        # The logger is now set up available globally via get_current_logger().
        # It will automatically close at the end of this block.
        logger.info(f"Configuration loaded successfully. Loaded_values: {config}")
        logger.info("Starting training process.",
                     extra={"status":"started", log_dir:"config_dir"})

        try:
            # Call the decorated functions(logger is automatically injected)
            train_model(config = config)  
           
        except Exception as e:
            logger.error(f"Critical error during training: {str(e)}", exc_info=True,
                           extra = {"status": "failed", "error": str(e)})
            raise
        
        finally:
            logger.info("Training process completed. Log file will be closed automatically.",
                         extra = {"status": "completed"})

        # Remove log files older than 30 days
        clean_old_logs(days=30) 


if __name__ == "__main__":
    main()
