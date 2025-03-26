import copy
import datetime
import functools
import logging
import time
import urllib.request
from threading import Lock, Thread
from typing import Optional, Union

logger = logging.getLogger(__name__)


class StatRecord:
    """Recording object that represents one or many stats measurements reported by download request threads.

    This object supports creation with a full set of measurements from one thread. It also supports combining
    instances of the class to create aggregate stats from several measurements.
    """

    # Base stats fields. Other stats are all derived from these.
    # All stats tracked must be part of this dict
    field_defaults = {
        "request_duration": datetime.timedelta(),  # Time from request sent to first byte from the server
        "response_duration": datetime.timedelta(),  # Total time spent recieving and processing a response
        "request_size_bytes": 0,  # Total size of all requests
        "response_size_bytes": 0,  # Total size of all responses
        "request_duration_avg": 0,  # Average request duration
        "snapshots": 0,  # Number of fits snapshots downloaded
    }

    def __init__(self, received_at: datetime.datetime, data_start: datetime.datetime, **kwargs):
        # Receipt time and number of data points are handled specially
        # by accumulation functions
        self.received_at = received_at
        self.data_start = data_start
        self.N = 1

        self.__dict__.update(StatRecord.field_defaults)
        self.__dict__.update(kwargs)

    @staticmethod
    def combine(obja: "StatRecord", objb: Optional["StatRecord"]) -> "StatRecord":
        """Combine two stats object into a new stats object representing the combination of the two.

        Stats objects can be the result of n processed data points, so when combining two objects
        with n and m data points, the new object will indicate it is composed of n+m data points.

        Parameters
        ----------
        obja : StatRecord
            The first stats record

        objb : StatRecord, optional
            The second stats record. If not provided a copy of obja will be returned.

        Returns
        -------
        StatRecord
            A new object with the combination of the two stats objects.
        """
        return copy.copy(obja) if objb is None else obja._combine(objb)

    def _combine(self, obj: "StatRecord") -> "StatRecord":
        """Combine this object with another. Private helper for combine()."""
        returnme = copy.copy(self)

        for key in StatRecord.field_defaults:
            returnme._stat_update(key, obj.__dict__[key], obj.N)

        # Update total number of data points
        returnme.N += obj.N

        # Pick the earliest moment for data_start
        returnme.data_start = min(self.data_start, obj.data_start)

        # Pick the latest moment for recieved_at
        returnme.received_at = max(self.received_at, obj.received_at)

        return returnme

    def _stat_update(self, key: str, value: Union[int, datetime.timedelta], m: int = 1):
        """Update a stat within the record object. Automatically averages if the stat key ends with _avg

        Parameters
        ----------
        key : str
            The stat key to update.
        value : Union[int, datetime.timedelta]
            The value to update it with.
        m : int, optional
            If we are averaging, we need to know how many data points the current average you are passing
            in represents, by default 1
        """
        if key[-4:] == "_avg":
            # Compute overall average given two averages of different numbers of data points
            old_avg = self.__dict__[key]
            n = self.N
            self.__dict__[key] = (n / (n + m)) * old_avg + (value / (n + m))
        else:
            # Simple accumulation
            self.__dict__[key] += value

    # Utility function to syntactic sugar away divisions-by-zero in stats code.
    @staticmethod
    def _div(num: float, denom: float, default: float = 0.0) -> float:
        return num / denom if denom != 0 else default

    def wall_clock_dur_s(self) -> int:
        """The wall clock duration that is represented by this stats object.
        How long between the beginning of when it was recording to the moment it was reported
        to the stats system in seconds?

        Returns
        -------
        int
            The number of seconds
        """
        return int((self.received_at - self.data_start).total_seconds())

    def total_dur_s(self) -> int:
        """Total duration of time that a thread was doing data transfer. In the case of multiple threads
        each thread's seconds are added.

        A stats object can represent an amalgamation of measurements from multiple worker threads.
        `total_dur_s` represents the total time data was being transferred added across all threads.

        Returns
        -------
        int
            The number of seconds
        """
        return int((self.request_duration + self.response_duration).total_seconds())  # type: ignore[attr-defined]

    def resp_s(self) -> int:
        """The time spent receiving responses from the server, added across all threads.

        Returns
        -------
        int
            A number of seconds
        """
        return int(self.response_duration.total_seconds())  # type: ignore[attr-defined]

    def data_down_mb(self) -> float:
        """The amount of data downloaded by all threads together.

        Returns
        -------
        float
            A flooating point number of 1024-based megabytes
        """
        return float(self.response_size_bytes) / (1024**2)  # type: ignore[attr-defined]

    def down_rate_mb_s(self) -> float:
        """The downstream data rate in megabytes per second experienced by the average thread.

        `data_down_mb`/`resp_s`.

        Returns
        -------
        float
            A floating point number of 1024-based megabytes per second
        """
        return StatRecord._div(self.data_down_mb(), self.resp_s())

    def down_rate_mb_s_overall(self) -> float:
        """The downstream data rate in megabytes per second expereineced by an aggregate of threads.

        `data_down_mb`/`wall_clock_dur_s`

        Returns
        -------
        float
            A floating point number of 1024-based megabytes per second
        """
        return StatRecord._div(self.data_down_mb(), self.wall_clock_dur_s())

    def req_s(self) -> int:
        """The time spent sending requests to the server, added across all threads.

        Returns
        -------
        int
            A number of seconds
        """
        return int(self.request_duration.total_seconds())  # type: ignore[attr-defined]

    def data_up_mb(self) -> float:
        """The amount of data uploaded by all threads together.

        Returns
        -------
        float
            A flooating point number of 1024-based megabytes
        """
        return float(self.request_size_bytes) / (1024**2)  # type: ignore[attr-defined]

    def up_rate_mb_s(self) -> float:
        """The upstream data rate in megabytes per second experienced by the average thread.

        `data_up_mb`/`resp_s`.

        Returns
        -------
        float
            A floating point number of 1024-based megabytes per second
        """
        return StatRecord._div(self.data_up_mb(), self.req_s())

    def snapshot_rate(self) -> float:
        """The rate of snapshots downloaded experienced by the average thread.

        (`snapshots`/`total_dur_s`) * 3600 * 24

        Returns
        -------
        float
            A floating point number of snapshot images per day
        """
        return StatRecord._div(self.snapshots, self.total_dur_s()) * 3600 * 24  # type: ignore[attr-defined]

    def snapshot_rate_overall(self) -> float:
        """The rate of snapshots downloaded by all threads together.

        ()`snapshots`/`wall_clock_dur_s`) * 3600 * 24

        Returns
        -------
        float
            A floating point number of snapshot images per day
        """
        return StatRecord._div(self.snapshots, self.wall_clock_dur_s()) * 3600 * 24  # type: ignore[attr-defined]

    def log_summary(self, log_level: int = logging.INFO, num_threads: int = 1, prefix: str = ""):
        """Log two lines of summary of this stats object at the given log level.

        The first line is an overall summary that treats all threads that gave the stats object data as
        a single unit

        The second line is a per-thread summary that produces thread-averaged statistics.

        If provided, prefix is appended to the beginning of the log lines emitted.

        Parameters
        ----------
        log_level : int
            The logging level to emit the log message at.
        num_threads : int
            Number of threads actively reporting into the stats object we want to print, by default 1
        prefix : str
            String to prefix all log messages with. "" by default
        """
        # Some things are calculated in print because we need the number of threads.

        # (Time threads spent doing data transfer) / (Time threads theoretically had to do data transfer)
        # 1.0 is unreachable, but represents a state where threads are always sending or recieving a request
        connnection_efficiency = StatRecord._div(self.total_dur_s(), self.wall_clock_dur_s() * num_threads)

        # On average how much time has a single thread been moving data
        thread_avg_dur = StatRecord._div(self.total_dur_s(), num_threads)

        stats_message = "Overall stats: "
        stats_message += f"Time: {self.wall_clock_dur_s():.2f} s, "
        stats_message += f"Files: {self.snapshots}, "  # type: ignore[attr-defined]
        stats_message += f"Download: {self.down_rate_mb_s_overall():.2f} MB/s, "
        stats_message += f"File rate: {self.snapshot_rate_overall():.0f} files/24h, "
        stats_message += f"Conn eff: {connnection_efficiency:.2f}"
        logger.log(log_level, prefix + stats_message)

        stats_message = f"Per-Thread Averages ({num_threads} threads): "
        stats_message += f"Time: {thread_avg_dur:.2f} s, "
        stats_message += f"Upload: {self.up_rate_mb_s():.2f} MB/s, "
        stats_message += f"Download: {self.down_rate_mb_s():.2f} MB/s, "
        stats_message += f"File rate: {self.snapshot_rate():.0f} files/24h, "
        logger.log(log_level, prefix + stats_message)


class DownloadStats:
    """Subsytem for keeping statistics on downloads. Used as a context manager in worker threads."""

    window_size = datetime.timedelta(hours=1)

    def __init__(self, print_interval_s=60):
        self.lock = Lock()

        self.cumulative_stats = None

        # List of stats dicts in the current window
        self.stats_window = []

        # Reference count active threads and whether printing has been started.
        self.active_threads = 0
        self.num_threads = 0
        self.print_stats = False

        # How often the watcher thread prints (seconds)
        self.print_interval_s = print_interval_s

        # Start our watcher thread to print stats to the log
        self.watcher_thread = Thread(
            target=self._watcher_thread, name="stats watcher thread", args=(logging.INFO,), daemon=True
        )
        self.watcher_thread.start()

    def __enter__(self):
        # Count how many threads are using stats
        with self.lock:
            self.active_threads += 1
            self.num_threads += 1

        return self.hook

    def __exit__(self, exc_type, exc_value, traceback):
        # Count how many threads are using stats
        with self.lock:
            self.active_threads -= 1

    def _watcher_thread(self, log_level):
        # Simple polling loop to print
        while self.active_threads != 0 or not self.print_stats:
            if self.print_stats:
                self._print_stats(log_level)
            time.sleep(self.print_interval_s)

    def _print_stats(self, log_level):
        with self.lock:
            current_window_stats = functools.reduce(StatRecord.combine, self.stats_window)
            current_window_stats.log_summary(log_level, self.num_threads, "Trailing 1hr: ")
            self.cumulative_stats.log_summary(log_level, self.num_threads, "Cumulative:   ")

    def hook(
        self,
        request: urllib.request.Request,
        request_start: datetime.datetime,
        response_start: datetime.datetime,
        response_size: int,
        chunk_size: int,
    ):
        """This hook is called on each chunk of snapshots downloaded.
        It is called immediately after the server has finished responding to the
        request, so datetime.datetime.now() is the end moment of the request

        Parameters
        ----------
        request : urllib.request.Request
            The request object relevant to this call
        request_start : datetime.datetime
            The moment the request was handed off to urllib.request.urlopen()
        response_start : datetime.datetime
            The moment there were bytes from the server to process
        response_size : int
            The size of the response from the server in bytes
        chunk_size : int
            The number of cutout files recieved in this request
        """

        now = datetime.datetime.now()
        request_duration = response_start - request_start
        response_duration = now - response_start
        if request.data is None:
            logger.info("Cannot determine the length of HTTP request. Ignoring this requests data in counts.")
            return
        request_size = len(request.data)  # type: ignore[arg-type]

        # Create a StatRecord for this report from a worker thread.
        current = StatRecord(
            now,
            request_start,
            **{
                "request_duration": request_duration,
                "response_duration": response_duration,
                "response_duration_avg": response_duration,
                "request_size_bytes": request_size,
                "response_size_bytes": response_size,
                "snapshots": chunk_size,
            },
        )

        with self.lock:
            # If this is the first piece of data we've received, signal that printing can start
            # on the next poll
            if not self.print_stats:
                self.print_stats = True

            # Combine our current stats into the cumulative stats
            self.cumulative_stats = StatRecord.combine(current, self.cumulative_stats)

            # Append the current stat record to the window
            self.stats_window.append(current)

            # Prune the window to the appropriate trailing time.
            window_threshold = now - DownloadStats.window_size
            self.stats_window = [rec for rec in self.stats_window if rec.received_at > window_threshold]
