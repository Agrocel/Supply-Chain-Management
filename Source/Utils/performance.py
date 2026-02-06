import time
from functools import wraps
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..')))
from Logging.logger import get_logger

logger = get_logger("Performance")

def measure_time(step_name):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            end = time.perf_counter()

            self = args[0]  # class instance (SCMPipeline, DataCleaner, etc.)

            # Optional: row count if df is passed
            rows = "NA"
            if len(args) > 1 and hasattr(args[1], "__len__"):
                try:
                    rows = len(args[1])
                except Exception:
                    pass

            self.logger.info(
                f"[PERF] {step_name} | rows={rows} | time={end - start:.4f}s"
            )
            return result
        return wrapper
    return decorator
