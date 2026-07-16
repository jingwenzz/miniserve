from miniserve.engine import Engine
from miniserve.executor import FakeExecutor
from miniserve.request import Request
from miniserve.scheduler import Scheduler


def format_request_ids(requests: list[Request]) -> list[str]:
    return [request.request_id for request in requests]


class VerboseEngine(Engine):
    def step(self) -> None:
        output = self.scheduler.schedule()

        print(
            f"step={self.step_count + 1} "
            f"prefill={format_request_ids(output.prefill_requests)} "
            f"decode={format_request_ids(output.decode_requests)}"
        )

        self.executor.prefill(output.prefill_requests)
        self.executor.decode(output.decode_requests)
        self.scheduler.update(output)

        print(
            f"         running={format_request_ids(self.scheduler.running)} "
            f"waiting={format_request_ids(list(self.scheduler.waiting))}"
        )

        self.step_count += 1


def main() -> None:
    scheduler = Scheduler(
        max_batch_size=3,
        max_batch_tokens=50,
    )
    executor = FakeExecutor(generated_token_id=42)
    engine = VerboseEngine(scheduler, executor)

    requests = [
        Request("request-1", [1] * 10, max_new_tokens=4),
        Request("request-2", [1] * 40, max_new_tokens=2),
        Request("request-3", [1] * 5, max_new_tokens=3),
    ]

    for request in requests:
        scheduler.add_request(request)

    engine.run_until_complete()

    print("\nCompleted requests:")

    for request in requests:
        print(
            f"{request.request_id}: "
            f"prompt={request.prompt_length}, "
            f"output={request.output_length}"
        )


if __name__ == "__main__":
    main()