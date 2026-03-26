# coding: utf-8

from src.core.utils.log_training_callbacks import LogTrainingCallbacks


def test_log_training_callbacks_full_cycle(mock_logger, caplog):
    """
    Test the full lifecycle of the training callback to reach 100% coverage.
    """
    # 1. Initialization
    callback = LogTrainingCallbacks(logger=mock_logger)

    # Keras typically sets a 'params' dict on the callback before training
    callback.params = {'epochs': 2, 'steps': 10}

    # 2. Test on_train_begin
    # This covers the pluralization logic and targeted epochs/steps printing
    callback.on_train_begin()

    # 3. Test Batch Cycle (Step 1)
    callback.on_train_batch_begin(batch=0)

    # Simulate a small delay or just call end
    callback.on_train_batch_end(batch=0)

    # 4. Test Batch Cycle (Step 2)
    callback.on_train_batch_begin(batch=1)
    callback.on_train_batch_end(batch=1)

    # 5. Test on_epoch_end
    # We provide a dummy logs dict as Keras would
    dummy_logs = {'loss': 0.45678, 'val_loss': 0.51234}
    callback.on_epoch_end(epoch=0, logs=dummy_logs)

    # --- ASSERTIONS ---
    # Verify the expected messages are in the captured logs
    assert "Epoch 1 finished" in caplog.text
    assert "loss: 0.4568" in caplog.text
    assert "val_loss: 0.5123" in caplog.text
