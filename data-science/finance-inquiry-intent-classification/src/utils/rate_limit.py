import time
from collections import deque


class SimpleRateLimiter:
    def __init__(self, max_calls, per_seconds):
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self.calls = deque()

    def acquire(self):
        now = time.time()

        while self.calls and self.calls[0] <= now - self.per_seconds:
            self.calls.popleft()

        if len(self.calls) >= self.max_calls:
            raise RuntimeError("Rate limit exceeded")

        self.calls.append(now)