import os
import time
from dataclasses import dataclass
from subprocess import Popen, PIPE
from threading import Thread


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
            gpus = getGPUs()
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


"""The following is based on GPUtil. It has been striped down to only the
functionality needed for Hyrax. The original code can be found here:
https://github.com/anderskm/gputil
"""
@dataclass
class GPU:
    """Holds the GPU metrics retrieved from nvidia-smi."""
    id: int
    load: float
    memoryTotal: float
    memoryUsed: float

    @property
    def memoryUtil(self):
        """Return the memory utilization of the GPU."""
        return self.memoryUsed / self.memoryTotal


def safeFloatCast(strNumber):
    try:
        number = float(strNumber)
    except ValueError:
        number = float('nan')
    return number


def getGPUs():
    """Get the GPU utilization and memory usage for all GPUs on the system using
    nvidia-smi. Returns a list of GPU objects."""

    try:
        p = Popen(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.total,memory.used", "--format=csv,noheader,nounits"], stdout=PIPE)
        stdout, stderror = p.communicate()
    except:
        return []
    output = stdout.decode('UTF-8')

    # Parse output
    lines = output.split(os.linesep)

    numDevices = len(lines)-1
    GPUs = []
    for g in range(numDevices):
        line = lines[g]
        vals = line.split(', ')
        for i in range(4):
            if (i == 0):
                deviceIds = int(vals[i])
            elif (i == 1):
                gpuUtil = safeFloatCast(vals[i])/100
            elif (i == 2):
                memTotal = safeFloatCast(vals[i])
            elif (i == 3):
                memUsed = safeFloatCast(vals[i])

        GPUs.append(GPU(deviceIds, gpuUtil, memTotal, memUsed))
    return GPUs
