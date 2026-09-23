from typing import Any


class Event:
    """
    Represents an event that can be triggered and listened to.
    """
    def __init__(self, name: str):
        self._listeners = []
        self._name = name

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
