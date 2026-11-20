"""Tests for streaming token events with incremental decoding.

These tests use scripted executors, so they need no model.
Two decoding hazards get their own tokenizers: one whose token
text depends on the whole sequence, where decoding each token
in isolation would produce a different string, and one where a
single token carries only half of a multi byte character, so
its isolated decoding is a replacement character that must
never reach the client as if it were the text.
"""

import threading
import time

from miniserve.engine import Engine
from miniserve.executor import FakeExecutor
from miniserve.request import Request
from miniserve.scheduler import Scheduler
from miniserve.service import EngineOwner
from miniserve.streaming import IncrementalDecoder, StreamEvent


class SequenceExecutor(FakeExecutor):
    """Fake executor appending a scripted token sequence per request."""

    def __init__(self, sequences: dict[str, list[int]]) -> None:
        super().__init__()
        self._sequences = {key: list(value) for key, value in sequences.items()}

    def decode(self, requests: list[Request]) -> None:
        for request in requests:
            token_id = self._sequences[request.request_id].pop(0)
            request.append_token(token_id)


class SequenceTokenizer:
    """Decode a whole sequence; isolated tokens decode differently.

    Token 2 alone decodes with a trailing marker, and token 4
    alone decodes as a replacement character because it is only
    the first half of a character that token 5 completes. Both
    are properties of real tokenizers (context dependent pieces
    and split multi byte characters), reduced to a table here.
    """

    def decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        table = {
            (1,): "hel",
            (2,): "lo!",
            (1, 2): "hello",
            (1, 2, 3): "hello world",
            (4,): "�",
            (4, 5): "café",
            (7,): " Paris",
            (7, 8): " Paris is",
        }
        return table[tuple(token_ids)]


def make_owner(
    executor: FakeExecutor, tokenizer: object | None = None
) -> EngineOwner:
    scheduler = Scheduler(max_batch_size=4, max_batch_tokens=100)
    engine = Engine(scheduler, executor)
    return EngineOwner(engine, tokenizer=tokenizer)


def collect(handle_events: object) -> list[StreamEvent]:
    return list(handle_events)  # type: ignore[arg-type]


def test_token_events_stream_in_order_with_one_terminal() -> None:
    executor = SequenceExecutor({"s": [7, 8]})
    owner = make_owner(executor, tokenizer=SequenceTokenizer())
    owner.start()
    try:
        handle = owner.submit(
            Request(request_id="s", prompt_token_ids=[1, 1], max_new_tokens=2)
        )
        events = collect(handle.events())
        assert [event.type for event in events] == ["token", "token", "finished"]
        assert [event.token_id for event in events[:2]] == [7, 8]
        assert [event.index for event in events[:2]] == [0, 1]
        terminal = events[-1]
        assert terminal.finish_reason == "finished"
        assert terminal.generated_tokens == 2
        assert handle.stream_text == " Paris is"
        sse = events[0].format_sse()
        assert sse.startswith("event: token\ndata: ")
        assert '"token_id": 7' in sse
    finally:
        owner.stop(timeout=5)


def test_incremental_decoding_uses_the_whole_sequence() -> None:
    executor = SequenceExecutor({"m": [1, 2, 3]})
    owner = make_owner(executor, tokenizer=SequenceTokenizer())
    owner.start()
    try:
        handle = owner.submit(
            Request(request_id="m", prompt_token_ids=[1], max_new_tokens=3)
        )
        events = collect(handle.events())
        token_texts = [event.text for event in events if event.type == "token"]
        # Decoding token 2 in isolation would have produced the
        # marker form; the streamed deltas never contain it, and
        # their concatenation is the full sequence decoding.
        assert token_texts == ["hel", "lo", " world"]
        assert "".join(token_texts) == "hello world"
        assert handle.stream_text == "hello world"
    finally:
        owner.stop(timeout=5)


def test_split_character_is_held_until_it_completes() -> None:
    executor = SequenceExecutor({"u": [4, 5]})
    owner = make_owner(executor, tokenizer=SequenceTokenizer())
    owner.start()
    try:
        handle = owner.submit(
            Request(request_id="u", prompt_token_ids=[1], max_new_tokens=2)
        )
        events = collect(handle.events())
        token_texts = [event.text for event in events if event.type == "token"]
        assert "�" not in "".join(token_texts)
        assert "".join(token_texts) == "café"
    finally:
        owner.stop(timeout=5)


def test_first_token_event_arrives_before_generation_ends() -> None:
    gate = threading.Event()
    entered = threading.Event()

    class GatedSequenceExecutor(SequenceExecutor):
        def decode(self, requests: list[Request]) -> None:
            # Gate the second decode, before it appends: by then
            # the first step has completed and its token event
            # has been emitted, while this generation is still
            # in progress on the worker.
            if requests and requests[0].output_length == 1:
                entered.set()
                assert gate.wait(timeout=5)
            super().decode(requests)

    executor = GatedSequenceExecutor({"g": [7, 8]})
    owner = make_owner(executor, tokenizer=SequenceTokenizer())
    owner.start()
    try:
        handle = owner.submit(
            Request(request_id="g", prompt_token_ids=[1], max_new_tokens=2)
        )
        stream = handle.events()
        first = next(stream)
        assert first.type == "token"
        assert first.token_id == 7
        assert entered.wait(timeout=5)
        assert not handle.done()
        gate.set()
        rest = collect(stream)
        assert [event.type for event in rest] == ["token", "finished"]
    finally:
        gate.set()
        owner.stop(timeout=5)


def test_two_streams_do_not_cross_and_each_ends_once() -> None:
    executor = SequenceExecutor({"x": [7, 8], "y": [1, 2, 3]})
    owner = make_owner(executor, tokenizer=SequenceTokenizer())
    owner.start()
    try:
        handle_x = owner.submit(
            Request(request_id="x", prompt_token_ids=[1], max_new_tokens=2)
        )
        handle_y = owner.submit(
            Request(request_id="y", prompt_token_ids=[1], max_new_tokens=3)
        )
        events_x = collect(handle_x.events())
        events_y = collect(handle_y.events())
        assert {event.request_id for event in events_x} == {"x"}
        assert {event.request_id for event in events_y} == {"y"}
        assert [e.type for e in events_x].count("finished") == 1
        assert [e.type for e in events_y].count("finished") == 1
        assert events_x[-1].is_terminal and events_y[-1].is_terminal
        assert handle_x.stream_text == " Paris is"
        assert handle_y.stream_text == "hello world"
    finally:
        owner.stop(timeout=5)


def test_cancelled_stream_ends_with_cancelled_terminal() -> None:
    gate = threading.Event()
    entered = threading.Event()

    class GatedSequenceExecutor(SequenceExecutor):
        def decode(self, requests: list[Request]) -> None:
            # Gate before the first token is appended, so the
            # cancel command can be queued while this step is
            # provably still in progress.
            if requests and requests[0].request_id == "c":
                entered.set()
                assert gate.wait(timeout=5)
            super().decode(requests)

    executor = GatedSequenceExecutor({"c": [7, 8]})
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=100)
    engine = Engine(scheduler, executor)
    owner = EngineOwner(engine, tokenizer=SequenceTokenizer())
    owner.start()
    cancelled: dict[str, object] = {}

    def cancel_c() -> None:
        cancelled["result"] = owner.cancel("c")

    thread = threading.Thread(target=cancel_c)
    try:
        handle = owner.submit(
            Request(request_id="c", prompt_token_ids=[1], max_new_tokens=2)
        )
        assert entered.wait(timeout=5)
        thread.start()
        deadline = time.monotonic() + 5
        while owner._commands.qsize() < 1:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        gate.set()
        thread.join(timeout=5)
        assert cancelled["result"] is True
        events = collect(handle.events())
        # The token from the completed step is streamed first,
        # then the single terminal event reports the cancel
        # that took effect at that step boundary.
        assert [event.type for event in events] == ["token", "finished"]
        assert events[-1].finish_reason == "cancelled"
        assert events[-1].generated_tokens == 1
        assert handle.request.output_length == 1
    finally:
        gate.set()
        if thread.is_alive():
            thread.join(timeout=5)
        owner.stop(timeout=5)


def test_failed_stream_ends_with_a_single_error_event() -> None:
    class FailingSequenceExecutor(SequenceExecutor):
        def decode(self, requests: list[Request]) -> None:
            for request in requests:
                request.fail("decode forward failed: RuntimeError: boom")
                raise RuntimeError("boom")
            super().decode(requests)

    executor = FailingSequenceExecutor({"f": [7]})
    owner = make_owner(executor, tokenizer=SequenceTokenizer())
    owner.start()
    try:
        handle = owner.submit(
            Request(request_id="f", prompt_token_ids=[1], max_new_tokens=2)
        )
        events = collect(handle.events())
        assert [event.type for event in events] == ["error"]
        assert events[0].error_code == "generation_failed"
        assert events[0].error_message is not None
        assert "boom" in events[0].error_message
    finally:
        owner.stop(timeout=5)


def test_decoder_without_tokenizer_emits_ids_and_no_text() -> None:
    decoder = IncrementalDecoder()
    assert decoder.feed(10) == ""
    assert decoder.feed(11) == ""
    assert decoder.flush() == ""
    assert decoder.emitted_text == ""


def test_events_iterator_stops_after_terminal() -> None:
    executor = SequenceExecutor({"once": [7]})
    owner = make_owner(executor, tokenizer=SequenceTokenizer())
    owner.start()
    try:
        handle = owner.submit(
            Request(request_id="once", prompt_token_ids=[1], max_new_tokens=1)
        )
        events = collect(handle.events())
        assert [event.type for event in events] == ["token", "finished"]
        assert events[-1].is_terminal
    finally:
        owner.stop(timeout=5)
