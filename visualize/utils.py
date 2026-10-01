from typing import Any


class Event:
    """
    Represents an event that can be triggered and listened to.
    """
    def __init__(self, name: str):
        self._listeners = []
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def forward(cls, other: "Event") -> "ForwardEvent":
        return ForwardEvent(other)

    def add_listener(self, listener: Any):
        self._listeners.append(listener)

    def remove_listener(self, listener: Any):
        self._listeners.remove(listener)

    def fire(self, *args, **kwargs):
        for listener in self._listeners:
            listener(*args, **kwargs)


class AnyEvent(Event):
    def __init__(self, name: str, events: list[Event]):
        super().__init__(name)
        self._events = events
        for event in events:
            event.add_listener(self._received)

    def _received(self, *args, **kwargs):
        self.fire(*args, **kwargs)


class ForwardEvent(Event):
    def __init__(self, other: Event):
        super().__init__(other.name)
        self._other = other
        self._other.add_listener(self._forward)

    def _forward(self, *args, **kwargs):
        return self.fire(*args, **kwargs)
