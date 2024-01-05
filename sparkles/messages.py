# Licensed under a 3-clause BSD style license - see LICENSE.rst

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Message:
    """Message class."""

    category: str
    text: str
    idx: int | None = None

    def __getitem__(self, key):
        return getattr(self, key)


class MessagesList(list[Message]):
    categories = ("all", "info", "caution", "warning", "critical", "none")

    def __eq__(self, other):
        if isinstance(other, str):
            return [msg for msg in self if msg["category"] == other]
        else:
            return super().__eq__(other)

    def __ge__(self, other):
        if isinstance(other, str):
            other_idx = self.categories.index(other)
            return [
                msg
                for msg in self
                if self.categories.index(msg["category"]) >= other_idx
            ]
        else:
            return super().__ge__(other)
