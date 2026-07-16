from miniserve.request import Request, RequestState

# prefill:
# WAITING → PREFILLING → DECODING

# decode:
# 每次调用，为每个 request 追加一个 token

class FakeExecutor:
    """Simulates prefill and decode without running a real model."""

    def __init__(self, generated_token_id: int = 1) -> None:
        self.generated_token_id = generated_token_id

    def prefill(self, requests: list[Request]) -> None:
        for request in requests:
            request.start_prefill()

        # A real executor would run the prompt through all Transformer layers
        # and initialize the KV cache here.

        for request in requests:
            request.finish_prefill()

    def decode(self, requests: list[Request]) -> None:
        for request in requests:
            if request.state is not RequestState.DECODING:
                raise RuntimeError(
                    f"Request {request.request_id} is not ready for decode"
                )

            request.append_token(self.generated_token_id)