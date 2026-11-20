"""Streaming events and incremental decoding for the service layer.

The event vocabulary is the one frozen in ``docs/service.md``:
zero or more ``token`` events, then exactly one terminal event,
either ``finished`` or ``error``. Nothing is emitted after the
terminal event, and the terminal event is never sent twice. The
engine owner in ``miniserve.service`` produces these events; this
module only defines their shape, their server sent events wire
format, and the incremental decoder that turns a growing token
sequence into text deltas.

Incremental decoding matters because a token id has no fixed
text of its own. A byte pair encoding piece can change meaning
with its neighbours, and one token can carry only part of a
multi byte character, so decoding each token in isolation and
concatenating the pieces is not always the generated text. The
decoder here instead re-decodes the whole sequence produced so
far and emits only the new suffix of the decoded text, which is
the standard prefix method: the concatenation of every emitted
delta equals the decoding of the full sequence. A trailing
replacement character is treated as unstable and held back,
because it usually means the newest token ended in the middle
of a character that a later token completes; the held part is
emitted once it becomes stable, or by ``flush`` at the end of
the request. A token whose delta is empty for that reason is
explicitly allowed by the service contract. Client receipt time
of an event is not an engine token time; timing stays in the
runtime metrics record.
"""

import json
from dataclasses import dataclass
from typing import Any

REPLACEMENT_CHARACTER = "�"


@dataclass(frozen=True)
class StreamEvent:
    """One event on a request stream, in the frozen vocabulary."""

    type: str
    request_id: str
    index: int | None = None
    token_id: int | None = None
    text: str = ""
    finish_reason: str | None = None
    generated_tokens: int | None = None
    error_code: str | None = None
    error_message: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.type in {"finished", "error"}

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON object carried as the event data."""
        data: dict[str, Any] = {
            "type": self.type,
            "request_id": self.request_id,
        }
        if self.type == "token":
            data["index"] = self.index
            data["token_id"] = self.token_id
            data["text"] = self.text
        elif self.type == "finished":
            data["finish_reason"] = self.finish_reason
            data["generated_tokens"] = self.generated_tokens
        elif self.type == "error":
            data["error"] = {
                "code": self.error_code,
                "message": self.error_message,
                "request_id": self.request_id,
            }
        return data

    def format_sse(self) -> str:
        """Return this event as one server sent events frame."""
        return f"event: {self.type}\ndata: {json.dumps(self.to_dict())}\n\n"


class IncrementalDecoder:
    """Turn a growing token sequence into stable text deltas.

    The tokenizer is any object with a ``decode`` method that
    maps a list of token ids to text, which is the transformers
    tokenizer interface. With no tokenizer, every delta is the
    empty string: token events still carry their ids, and no
    text is invented for a backend that has none.
    """

    def __init__(self, tokenizer: Any | None = None) -> None:
        self._tokenizer = tokenizer
        self._token_ids: list[int] = []
        self._emitted = ""

    @property
    def emitted_text(self) -> str:
        """The concatenation of every delta returned so far."""
        return self._emitted

    def _decode_all(self) -> str:
        if self._tokenizer is None or not self._token_ids:
            return ""
        return str(self._tokenizer.decode(list(self._token_ids)))

    def feed(self, token_id: int) -> str:
        """Add one token and return its stable text delta."""
        self._token_ids.append(token_id)
        if self._tokenizer is None:
            return ""
        decoded = self._decode_all()
        stable = decoded
        while stable.endswith(REPLACEMENT_CHARACTER):
            stable = stable[: -len(REPLACEMENT_CHARACTER)]
        if not stable.startswith(self._emitted):
            # The decoder rewrote an earlier part of the text, so
            # no suffix of this decoding can be emitted honestly
            # yet. Hold everything; flush settles the final text.
            return ""
        delta = stable[len(self._emitted) :]
        self._emitted = stable
        return delta

    def flush(self) -> str:
        """Return any held text once no more tokens will come."""
        if self._tokenizer is None:
            return ""
        decoded = self._decode_all()
        if not decoded.startswith(self._emitted):
            return ""
        delta = decoded[len(self._emitted) :]
        self._emitted = decoded
        return delta
