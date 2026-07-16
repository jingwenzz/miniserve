from miniserve.executor import FakeExecutor
from miniserve.scheduler import Scheduler

# schedule
#    ↓
# prefill 新请求
#    ↓
# decode 原有请求
#    ↓
# update queues

class Engine:
    """Orchestrates scheduling and execution."""

    def __init__(
        self,
        scheduler: Scheduler,
        executor: FakeExecutor,
    ) -> None:
        self.scheduler = scheduler
        self.executor = executor
        self.step_count = 0

    def step(self) -> None:
        output = self.scheduler.schedule()

        self.executor.prefill(output.prefill_requests)
        self.executor.decode(output.decode_requests)

        self.scheduler.update(output)
        self.step_count += 1

    def run_until_complete(self) -> None:
        while self.scheduler.has_pending_requests():
            self.step()