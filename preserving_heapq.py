import heapq
from typing import *

T = TypeVar("T")
S = TypeVar("S", int, float)


class NoPriorityFunctionExecption(Exception):
    pass


class PreservingHeapQueue(Generic[T, S]):
    f: Callable[[T], Optional[S]]
    counter: int
    queue: List[Tuple[S, int, T]]

    def __init__(self, f: Optional[Callable[[T], S]] = None, init: List[T] = None):
        if init is None:
            init = []
        self.counter = 0
        self.queue = []

        def assertion(_: T):
            raise NoPriorityFunctionExecption()

        if f is None:
            self.f = assertion
        else:
            self.f = f
        for i in init:
            self.push(i)

    def push(self, x: T) -> S:
        priority = self.f(x)
        self.push_with_priority(x, priority)
        return priority

    def push_with_priority(self, x: T, p: S):
        heapq.heappush(self.queue, (p, self.counter, x))
        self.counter += 1

    def pop(self) -> Optional[Tuple[T, S]]:
        if self.queue:
            priority, _, x = heapq.heappop(self.queue)
            return x, priority
        else:
            return None

    def to_list(self) -> List[T]:
        return [x for _, _, x in self.queue]

    def is_empty(self) -> bool:
        if self.queue:
            return False
        else:
            return True

    def __len__(self) -> int:
        return len(self.queue)

    def __str__(self):
        return str(self.to_list())
