import time
from threading import Thread

import GPUtil


class GpuMonitor(Thread):
    """General GPU monitor that runs in a separate thread and logs GPU metrics
    to Tensorboard.
    """

    def __init__(self, tensorboard_logger, interval_seconds=1):
        super().__init__()
        self.stopped = False
        self.delay = interval_seconds  # Seconds between calls to GPUtil
        self.start_time = time.time()
        self.tensorboard_logger = tensorboard_logger
        self.start()

    def run(self):
        """Run loop that logs GPU metrics every `self.delay` seconds."""
        while not self.stopped:
            gpus = GPUtil.getGPUs()
            step = time.time() - self.start_time
            for gpu in gpus:
                gpu_name = f"GPU_{gpu.id}"
                self.tensorboard_logger.add_scalar(f"{gpu_name}/load", gpu.load * 100, step)
                self.tensorboard_logger.add_scalar(
                    f"{gpu_name}/memory_utilization", gpu.memoryUtil * 100, step
                )
            time.sleep(self.delay)

    def stop(self):
        """Stop the monitoring thread."""
        self.stopped = True
