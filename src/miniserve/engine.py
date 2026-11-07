from miniserve.executor import Executor
from miniserve.request import Request
from miniserve.scheduler import Scheduler

# schedule
#    ↓
# prefill 新请求
#    ↓
# decode 原有请求
#    ↓
# update queues
#    ↓
# release backend resources for requests that reached a terminal state


class Engine:
    """Orchestrates scheduling and execution.

    The engine depends on the Executor protocol, not on any concrete
    backend. It drives one step as schedule, prefill, decode, update,
    and it owns the timing of executor release calls: state changes
    during execution are the executor's job, queue changes are the
    scheduler's job, and the engine does not duplicate either one.

    The engine tracks every request the executor has started work on
    so a request that is cancelled between steps, while it is sitting
    in the running list, still gets released. A request cancelled
    while waiting was never tracked and holds no backend resources,
    so it needs no release.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        executor: Executor,
    ) -> None:
        self.scheduler = scheduler
        self.executor = executor
        self.step_count = 0
        self._tracked: dict[str, Request] = {}
        self._released: set[str] = set()

    def _track(self, requests: list[Request]) -> None:
        for request in requests:
            self._tracked[request.request_id] = request

    def release_terminal_requests(self) -> None:
        """Release tracked requests that reached a terminal state.

        Each tracked request is released at most once from the
        engine side. The executor contract also requires release to
        be idempotent on the executor side, as a second safety net.
        """
        for request_id, request in list(self._tracked.items()):
            if request.is_finished and request_id not in self._released:
                self.executor.release(request)
                self._released.add(request_id)

    def step(self) -> None:
        output = self.scheduler.schedule()

        self._track(output.prefill_requests)
        self._track(output.decode_requests)

        self.executor.prefill(output.prefill_requests)
        self.executor.decode(output.decode_requests)

        self.scheduler.update(output)
        self.release_terminal_requests()
        self.step_count += 1

    def run_until_complete(self) -> None:
        while self.scheduler.has_pending_requests():
            self.step()
        # A request cancelled between steps may leave the queues
        # without another step running, so sweep once more here.
        self.release_terminal_requests()
