# MiniServe Service Boundary: Minimal API Contract

This document fixes the smallest service boundary around the
verified runtime, before any server code exists. It defines one
generation entry point, one health check, one cancellation
command, the event vocabulary for streaming, and the error
responses for invalid input, a full queue, and an execution
failure. The implementation lands in later steps against this
contract; nothing in this document claims a running server yet.

## Principles

- The model is loaded exactly once, at service startup, from
  the locked configuration in `configs/model.json`, and shared
  by every request. No request path loads, reloads, or swaps a
  model.
- The engine has a single owner. The API layer never touches
  the scheduler, the executor, or a request object directly; it
  hands commands (admit this prompt, cancel this id) to the
  engine owner and translates the events that come back. How
  the owner runs the loop is the next implementation step, and
  it does not change this contract.
- The API layer owns only four things: parsing and shape
  validation, assigning the request id, translating runtime
  outcomes into the wire formats below, and carrying events to
  the client. Admission limits, scheduling, execution,
  cancellation timing, and metrics stay in the runtime, exactly
  where they are already tested.
- Decoding is greedy, the model and revision are the locked
  ones, and there is one model per process. The contract is
  deliberately not OpenAI compatible: field names follow this
  project's own request and metrics vocabulary, and no
  compatibility is promised or implied.
- A client visible time is not an engine time. Streaming
  events carry no timing claims; per request timing lives in
  the runtime metrics record, whose definitions are frozen in
  `docs/metrics.md`.

## Endpoints

| Method and path | Purpose | Returns |
| --- | --- | --- |
| `GET /health` | Liveness and identity of the loaded model | Health object |
| `POST /generate` | Run one prompt to completion | Result object, or one error response |
| `POST /generate/stream` | Run one prompt, events as they happen | Server sent events: `token`, then exactly one of `finished` or `error` |
| `POST /requests/{request_id}/cancel` | Ask the runtime to cancel one admitted request | Cancellation outcome object |

The streaming transport is implemented in its own step; the
event vocabulary below is fixed now so that step cannot invent
a second outcome language. Cancellation propagation from a
disconnected client is likewise a later step; the command and
its responses are fixed here.

## Generation request

```json
{
  "prompt": "The capital of France is",
  "max_new_tokens": 16
}
```

| Field | Type | Rule |
| --- | --- | --- |
| `prompt` | string | Required, must not be empty. Token level checks (prompt length against the step budget and the context limit) are runtime admission checks, not API checks, because only the runtime owns the tokenizer counts and the limits. |
| `max_new_tokens` | integer | Optional, defaults to 16. Must be a positive integer when present. |

Unknown fields are ignored. There is no sampling field:
decoding is greedy by project rule, and a request that asks
for sampling options is served greedily, not rejected.

## Generation result

```json
{
  "request_id": "req-9f2c1a",
  "text": " Paris, the capital of France, is",
  "finish_reason": "finished",
  "prompt_tokens": 5,
  "generated_tokens": 8
}
```

- `request_id` is assigned by the service at admission and is
  the id the runtime knows the request by, so a cancellation
  or a metrics lookup names the same request.
- `finish_reason` uses the runtime vocabulary from
  `docs/metrics.md`: `finished` for a request that reached its
  output limit or produced the EOS token. A request that ends
  `cancelled` or `failed` never appears as a successful result
  object; cancellation is reported by the cancel call and by
  the terminal stream event, and failure is an error response.
- `text` is the decoded generated text. Token ids are not part
  of the non streaming result; they are available in the
  runtime metrics record for debugging.

## Streaming events

Each event is one server sent event whose data is a JSON
object with a `type` field:

```text
event: token
data: {"type": "token", "request_id": "req-9f2c1a", "index": 0, "token_id": 260, "text": " Paris"}

event: finished
data: {"type": "finished", "request_id": "req-9f2c1a", "finish_reason": "finished", "generated_tokens": 8}
```

- `token` events carry the token id and the incremental text
  for that token. Incremental text comes from incremental
  decoding on the server side; a token whose bytes do not yet
  form a complete character may carry an empty `text`, and the
  concatenation of all `text` fields equals the result `text`.
  Client receipt time of a chunk is not the engine first token
  time.
- Exactly one terminal event ends the stream: `finished` as
  above, or `error` with the error object below. No event is
  sent after the terminal event, and the terminal event is
  never sent twice.
- A cancelled request ends its stream with a `finished` event
  whose `finish_reason` is `cancelled` and whose
  `generated_tokens` counts the tokens produced before the
  cancellation took effect at its scheduling boundary.

## Cancellation

`POST /requests/{request_id}/cancel` mirrors the scheduler
cancellation contract one to one:

```json
{"request_id": "req-9f2c1a", "cancelled": true}
```

- `cancelled` is `true` when an active waiting or running
  request was cancelled, and `false` when the request had
  already reached a terminal state, in which case the response
  also carries its `finish_reason`. Cancelling twice is safe.
- An id the service never admitted is answered with the
  `unknown_request` error below, matching the runtime rule
  that an unknown id is an error, not a silent no-op.
- Cancellation takes effect at the next scheduling boundary,
  not inside a forward that is already running; the contract
  states this plainly so no client expects an in flight token
  to be retracted.

## Errors

Every error response, from either generation path or the
cancel call, has the same shape:

```json
{
  "error": {
    "code": "queue_full",
    "message": "The waiting queue is full. Retry later.",
    "request_id": null
  }
}
```

`request_id` is the admitted request's id when one exists, and
`null` when the request was never admitted.

| HTTP status | Code | When | Runtime source |
| --- | --- | --- | --- |
| 400 | `invalid_request` | Body is not a JSON object, `prompt` is missing, empty, or not a string, or `max_new_tokens` is present and not a positive integer. Raised by API validation; no runtime request is created and no id is consumed. | API layer |
| 400 | `request_rejected` | The runtime refused admission because the request can never run: prompt longer than the per step token budget, prompt plus reserved output over the context limit, or reservation over the cache capacity on its own. The `message` carries the scheduler's reason text. | `RequestRejectedError` |
| 429 | `queue_full` | The waiting queue is at its configured bound, so the request was not admitted and not queued. Nothing is silently dropped: the client is told at once and may retry later. The bound itself is service configuration set with the overload work; this contract fixes the response. | Admission bound |
| 500 | `generation_failed` | The executor failed after admission. The request ends in the runtime FAILED state with its reason recorded; the `message` carries that failure reason, and tokens produced before the failure stay in the metrics record. On a stream, this arrives as the terminal `error` event instead of a second response. | Executor failure path |
| 404 | `unknown_request` | A cancel call names an id the service never admitted. | Scheduler `KeyError` |

## Health

```json
{
  "status": "ok",
  "model_id": "HuggingFaceTB/SmolLM2-135M",
  "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
  "accepting_requests": true
}
```

`GET /health` answers 200 with this object once the model is
loaded and the engine owner is running. Before loading
finishes, or after a shutdown begins, it answers 503 with
`status` set to `unavailable` and `accepting_requests` set to
false. Health reports the service, not a generation: it runs
no forward and reserves no capacity.

## What the API layer must never do

- Run a model forward, pack a cache, or pick a schedule. If a
  handler can reach the executor, the boundary has failed.
- Invent an outcome. Empty text is not a failure report, a
  rejected request is not a queued one, and a failed request
  is never rewritten as `finished`.
- Hold per request generation state of its own. The runtime
  request, identified by `request_id`, is the only record of
  what a generation produced.

## Staged implementation, in order

1. Background execution with a single engine owner and a
   thread safe command queue, so a second request can be
   received while a first one is generating.
2. Streaming transport for the event vocabulary above.
3. Client disconnect mapped to the cancel command.
4. The waiting queue bound and the slow consumer policy that
   the `queue_full` response assumes.
5. An integration pass over two normal clients, one
   cancellation, one invalid request, and one overload case.
6. A freeze review that lists what the service does not
   support, starting with: no authentication, no sampling
   controls, no multi model serving, no OpenAI compatibility.
