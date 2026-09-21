from typing import Any
from abc import ABC


class ListVar:
    def __init__(self, initial: list | None = None):
        self._list = initial.copy() if initial is not None else []
        self.on_append = Event("on_append")
        self.on_remove = Event("on_remove")
        self.on_setitem = Event("on_setitem")
        self.on_clear = Event("on_clear")
        self.on_changed = AnyEvent("on_changed", [self.on_append, self.on_remove, self.on_setitem, self.on_clear])

    def append(self, value):
        self._list.append(value)
        self.on_append.fire(self, value)

    def remove(self, value):
        self._list.remove(value)
        self.on_remove.fire(self, value)

    def clear(self):
        self._list.clear()
        self.on_clear.fire(self)

    def __getitem__(self, key):
        return self._list[key]

    def __setitem__(self, key, value):
        self._list[key] = value
        self.on_changed.fire(self)

    def __len__(self):
        return len(self._list)


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