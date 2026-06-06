"""
YOLO Trainer — Callbacks
Real-time log capture for bridging Ultralytics training events to the Gradio UI.
Captures epoch-level metrics and training log messages via a thread-safe queue.
"""

import queue
import logging
from typing import Callable


def create_log_callback(log_queue: queue.Queue) -> Callable:
    """Create an Ultralytics-compatible callback that pushes training events to a queue.

    Events pushed to the queue:
        {"type": "epoch", "epoch": int, "total": int, "metrics": dict}
        {"type": "log", "message": str}
        {"type": "start", "total_epochs": int}
        {"type": "end", "reason": str, "final_metrics": dict}
        {"type": "error", "message": str}
    """

    def on_pretrain_routine_end(trainer):
        """Called after pre-training setup; signals the UI that training is starting."""
        total_epochs = getattr(trainer, "epochs", 0)
        log_queue.put({"type": "start", "total_epochs": total_epochs})

    def on_fit_epoch_end(trainer):
        """Called at the end of each training epoch."""
        epoch = getattr(trainer, "epoch", -1) + 1
        total = getattr(trainer, "epochs", 0)
        metrics = {}

        # Extract metrics from trainer
        if hasattr(trainer, "metrics") and trainer.metrics:
            for key, value in trainer.metrics.items():
                try:
                    if hasattr(value, "item"):
                        metrics[key] = round(value.item(), 6)
                    elif isinstance(value, (int, float)):
                        metrics[key] = round(float(value), 6)
                except Exception:
                    pass

        # Also check for results dict
        if hasattr(trainer, "validator") and hasattr(trainer.validator, "metrics"):
            val_metrics = trainer.validator.metrics
            if val_metrics:
                for key, value in val_metrics.items():
                    try:
                        k = f"val/{key}"
                        if hasattr(value, "item"):
                            metrics[k] = round(value.item(), 6)
                        elif isinstance(value, (int, float)):
                            metrics[k] = round(float(value), 6)
                    except Exception:
                        pass

        log_queue.put({
            "type": "epoch",
            "epoch": epoch,
            "total": total,
            "metrics": metrics,
        })

    def on_train_end(trainer):
        """Called when training completes."""
        metrics = {}
        if hasattr(trainer, "metrics") and trainer.metrics:
            for key, value in trainer.metrics.items():
                try:
                    if hasattr(value, "item"):
                        metrics[key] = round(value.item(), 6)
                    elif isinstance(value, (int, float)):
                        metrics[key] = round(float(value), 6)
                except Exception:
                    pass
        log_queue.put({"type": "end", "reason": "completed", "final_metrics": metrics})

    def on_train_epoch_end(trainer):
        """Same as on_fit_epoch_end; provides compatibility fallback."""
        # Ultralytics may call one or the other depending on version
        pass

    # Return a dict of callbacks (Ultralytics format)
    return {
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_train_end": on_train_end,
    }


class LogCaptureHandler(logging.Handler):
    """Custom logging handler that captures Ultralytics log messages into a queue."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
        self.setLevel(logging.INFO)

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.log_queue.put({"type": "log", "message": msg})
        except Exception:
            pass


def setup_log_capture(log_queue: queue.Queue) -> LogCaptureHandler:
    """Attach the queue-based log handler to Ultralytics' logger.

    Returns the handler so it can be removed after training.
    """
    handler = LogCaptureHandler(log_queue)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                           datefmt="%H:%M:%S"))

    # Attach to ultralytics logger
    ultralytics_logger = logging.getLogger("ultralytics")
    ultralytics_logger.addHandler(handler)

    return handler


def remove_log_capture(handler: LogCaptureHandler):
    """Remove the log capture handler from Ultralytics' logger."""
    if handler:
        logging.getLogger("ultralytics").removeHandler(handler)
