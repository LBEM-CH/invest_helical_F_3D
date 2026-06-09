#!/usr/bin/env python3
"""
Marked-for-removal store for invest_helical_F_3D.

author: Wen-Lu Chung

Holds the set of Dynamo tags (col 0) the user has marked to remove. It autosaves
to a plain text file (one tag per line, sorted) on every change -- that file is
BOTH the resume/progress list and the final export the user feeds to their own
.tbl cleanup. Emits a Qt signal so every open view restyles in lockstep.
"""

from __future__ import annotations

import os

from PyQt6 import QtCore


class SelectionStore(QtCore.QObject):
    changed = QtCore.pyqtSignal()        # any add/remove/clear/load

    def __init__(self, out_path: str, autosave: bool = True):
        super().__init__()
        self.out_path = out_path
        self.autosave = autosave
        self._marked: set[int] = set()
        if os.path.exists(out_path):
            self.load(out_path)          # resume where we left off

    # --- queries ---------------------------------------------------------
    def __contains__(self, tag: int) -> bool:
        return int(tag) in self._marked

    def is_marked(self, tag: int) -> bool:
        return int(tag) in self._marked

    def count(self) -> int:
        return len(self._marked)

    def tags(self) -> list[int]:
        return sorted(self._marked)

    def any_in(self, tags) -> bool:
        return any(int(t) in self._marked for t in tags)

    # --- mutations (each saves + signals once) ---------------------------
    def add(self, tags) -> None:
        self._marked.update(int(t) for t in _iter(tags))
        self._after_change()

    def remove(self, tags) -> None:
        for t in _iter(tags):
            self._marked.discard(int(t))
        self._after_change()

    def toggle(self, tag: int) -> None:
        tag = int(tag)
        if tag in self._marked:
            self._marked.discard(tag)
        else:
            self._marked.add(tag)
        self._after_change()

    def clear(self, tags=None) -> None:
        """Clear everything, or just the given tags."""
        if tags is None:
            self._marked.clear()
        else:
            for t in _iter(tags):
                self._marked.discard(int(t))
        self._after_change()

    # --- io --------------------------------------------------------------
    def load(self, path: str) -> None:
        """Load a tag list (ignores blanks / comments), replacing the current set."""
        marked: set[int] = set()
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                marked.add(int(float(line.split()[0])))
        self._marked = marked
        self._after_change()

    def save(self, path: str | None = None) -> None:
        path = path or self.out_path
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write("# invest_helical_F_3D remove list: Dynamo tags (col 1) to remove\n")
            for tag in sorted(self._marked):
                fh.write(f"{tag}\n")
        os.replace(tmp, path)            # atomic; safe if the app is killed mid-write

    def _after_change(self) -> None:
        if self.autosave:
            self.save()
        self.changed.emit()


def _iter(tags):
    if isinstance(tags, (int, float)):
        return [tags]
    return tags
