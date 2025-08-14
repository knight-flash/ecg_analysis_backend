import time
from collections import deque
from threading import Lock

class RequestController:
    """
    一个简单的、线程安全的速率控制器。
    使用滑动窗口算法来确保在指定时间窗口内请求不会超过最大次数。
    """
    def __init__(self, max_requests: int, per_seconds: int):
        """
        初始化速率控制器。
        Args:
            max_requests (int): 在时间窗口内允许的最大请求数。
            per_seconds (int): 时间窗口的长度（秒）。
        """
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.requests = deque()  # 使用双端队列存储最近的请求时间戳
        self.lock = Lock()       # 使用锁来保证线程安全

    def wait_for_slot(self):
        """
        等待直到有一个可用的请求槽位。
        """
        while True:
            with self.lock:
                current_time = time.time()

                # 1. 移除时间窗口之外的旧时间戳
                while self.requests and current_time - self.requests[0] > self.per_seconds:
                    self.requests.popleft()

                # 2. 检查当前窗口内的请求数是否已满
                if len(self.requests) < self.max_requests:
                    # 未满，记录当前请求时间戳并立即返回
                    self.requests.append(current_time)
                    print(f"[{time.strftime('%H:%M:%S')}] Request approved. Current count: {len(self.requests)}/{self.max_requests}")
                    return

                # 3. 已满，计算需要等待的时间
                # 等待时间 = 最早的那个请求过期所需的时间
                earliest_request_time = self.requests[0]
                time_to_wait = self.per_seconds - (current_time - earliest_request_time)

            print(f"[{time.strftime('%H:%M:%S')}] Rate limit reached. Waiting for {time_to_wait:.2f} seconds...")
            time.sleep(time_to_wait)