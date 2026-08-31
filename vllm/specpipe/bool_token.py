# SPDX-License-Identifier: Apache-2.0
"""Boolean sentinel passed between pipeline stages via Ray DAG channels.

Non-final pipeline stages normally emit ``IntermediateTensors`` into the
compiled-DAG channel. The Spexis target-model stage instead emits a
:class:`BoolToken` when the receiver must be told "no tensor payload this
step" (e.g. the hidden states were cached locally for the exit layer).
The class mirrors the small mapping interface of ``IntermediateTensors``
so both payload types can be handled uniformly by the channel plumbing.
"""

from dataclasses import dataclass


@dataclass
class BoolToken:
    """A tiny str -> bool mapping payload (e.g. ``{"tok": True}``)."""

    value: dict[str, bool]

    def __init__(self, value: dict[str, bool]):
        # Defined manually so Dynamo can attribute BoolToken() to this
        # file (a generated dataclass __init__ loses the source info).
        self.value = value

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.value[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.value.items()})

    def __setitem__(self, key: str, value: bool):
        self.value[key] = value

    def items(self):
        return self.value.items()

    def __len__(self):
        return len(self.value)

    def __repr__(self) -> str:
        return f"BoolToken({self.value})"
