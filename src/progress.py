"""Wall-clock reporting for the phases of the bulk load.

The load is an unattended setup step that runs for hours, so every phase has to
say how long it took and — wherever a throughput can be measured — how much
longer it has to go. Estimates are always extrapolated from the rate observed in
the current run rather than from stored constants, so they are correct on
whatever hardware the load happens to be running on.

Imported by both `loader.py` and `embeddings.py`; `loader.py` imports
`embeddings`, so this cannot live in either one.
"""

import time
from contextlib import contextmanager


def fmt_duration(seconds: float) -> str:
    """A compact duration: '41s', '1m18s', '11h42m'.

    One helper for a range spanning a 20-second ANALYZE and a 12-hour embed, so
    the unit never has to be read off a float ('~0.0 h left' was the old
    formatting's answer for anything under three minutes).
    """
    # Rounded before the branch, so 59.6s reads "1m00s" and not "60s".
    total = max(0, round(seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total // 3600}h{total % 3600 // 60:02d}m"


def eta(done: int, total: int, elapsed: float) -> str:
    """'(71/s, ~11h42m left)' from the rate measured so far.

    Returns "" until there is enough of a sample to divide by, so callers can
    concatenate it unconditionally rather than guarding every call site.
    """
    if done <= 0 or elapsed <= 0:
        return ""
    rate = done / elapsed
    return f"({rate:,.0f}/s, ~{fmt_duration((total - done) / rate)} left)"


class Progress:
    """Numbered phases with elapsed time, and timed steps inside them.

    Constructed with the labels of the phases this particular run will execute —
    which depend on the command-line flags — so the '[2/4]' denominator is
    honest instead of counting phases that were skipped.
    """

    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.done = 0

    @contextmanager
    def phase(self, label: str):
        self.done += 1
        print(f"\n[{self.done}/{len(self.labels)}] {label}", flush=True)
        started = time.monotonic()
        try:
            yield
        finally:
            # In a finally, so an interrupted phase still reports how long it
            # ran — the number is most wanted exactly when something went wrong.
            print(
                f"  -> {label} finished in {fmt_duration(time.monotonic() - started)}",
                flush=True,
            )

    @contextmanager
    def step(self, label: str):
        """Time one statement, printing its duration on the same line."""
        print(f"  {label} ...", end="", flush=True)
        started = time.monotonic()
        try:
            yield
        finally:
            print(f" {fmt_duration(time.monotonic() - started)}", flush=True)
