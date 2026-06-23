#!/usr/bin/env python3
"""
Marked-for-removal store for invest_helical_F_3D.

author: Wen-Lu Chung

Holds the set of Dynamo tags (col 0) the user has marked to remove. It autosaves
to a plain text file (one tag per line, sorted) on every change -- that file is
BOTH the resume/progress list and the final export the user feeds to their own
.tbl cleanup. Emits a Qt signal so every open view restyles in lockstep.

A second, independent channel holds FLIPPED segments: tags the user re-flipped to
the correct register, mapped to their new ZXZ-extrinsic angles (position
unchanged). It autosaves to flipped_list.txt next to the remove list. A tag may be
in both channels (the two are independent).
"""

from __future__ import annotations

import os

from PyQt6 import QtCore


class SelectionStore(QtCore.QObject):
    changed = QtCore.pyqtSignal()        # any add/remove/clear/load (either channel)

    def __init__(self, out_path: str, autosave: bool = True):
        super().__init__()
        self.out_path = out_path
        self.flip_path = os.path.join(os.path.dirname(out_path) or ".", "flipped_list.txt")
        self.autosave = autosave
        self._marked: set[int] = set()
        self._flips: dict[int, tuple] = {}      # tag -> (tdrot, tilt, narot) new angles
        if os.path.exists(out_path):
            self.load(out_path)          # resume where we left off
        if os.path.exists(self.flip_path):
            self.load_flips(self.flip_path)

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

    def set_all(self, tags) -> None:
        """Replace the whole marked set in one save+signal (used by invert-selection)."""
        self._marked = set(int(t) for t in _iter(tags))
        self._after_change()

    # --- flip channel (independent of removal) ---------------------------
    def is_flipped(self, tag: int) -> bool:
        return int(tag) in self._flips

    def flip_count(self) -> int:
        return len(self._flips)

    def flip_tags(self) -> list[int]:
        return sorted(self._flips)

    def get_flip(self, tag: int):
        """Resulting (tdrot, tilt, narot) angles for a flipped tag, or None."""
        v = self._flips.get(int(tag))
        return None if v is None else (v[2], v[3], v[4])

    def get_state(self, tag: int):
        """(tilt_flip, rot_flip) bits for a tag -- (0, 0) if not flipped."""
        v = self._flips.get(int(tag))
        return (0, 0) if v is None else (v[0], v[1])

    @staticmethod
    def _store_val(rec):
        """rec = (tilt, rot, (a1, a2, a3))  ->  (tilt, rot, a1, a2, a3)."""
        tilt, rot, ang = rec
        return (int(tilt), int(rot), float(ang[0]), float(ang[1]), float(ang[2]))

    def set_flips(self, mapping) -> None:
        """Add/replace flips. `mapping` is {tag: (tilt, rot, (a1, a2, a3))}; a
        (0, 0) state removes the tag."""
        for tag, rec in dict(mapping).items():
            if int(rec[0]) or int(rec[1]):
                self._flips[int(tag)] = self._store_val(rec)
            else:
                self._flips.pop(int(tag), None)
        self._after_flip_change()

    def unflip(self, tags) -> None:
        for t in _iter(tags):
            self._flips.pop(int(t), None)
        self._after_flip_change()

    def replace_flips(self, set_map, clear_tags=()) -> None:
        """Set/replace some flips and clear others in one go (single save+signal).
        set_map is {tag: (tilt, rot, (a1, a2, a3))}; a (0, 0) state is dropped."""
        for t in _iter(clear_tags):
            self._flips.pop(int(t), None)
        for tag, rec in dict(set_map).items():
            if int(rec[0]) or int(rec[1]):
                self._flips[int(tag)] = self._store_val(rec)
            else:
                self._flips.pop(int(tag), None)
        self._after_flip_change()

    def clear_flips(self, tags=None) -> None:
        if tags is None:
            self._flips.clear()
        else:
            for t in _iter(tags):
                self._flips.pop(int(t), None)
        self._after_flip_change()

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

    def load_flips(self, path: str) -> None:
        """Load 'tag tdrot tilt narot tilt_flip rot_flip' per line (comments ok). Old
        4-column files (no flip-type columns) are read as a tilt flip (1 0)."""
        flips: dict[int, tuple] = {}
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split()
                if len(p) >= 6:
                    flips[int(float(p[0]))] = (int(float(p[4])), int(float(p[5])),
                                               float(p[1]), float(p[2]), float(p[3]))
                elif len(p) >= 4:
                    flips[int(float(p[0]))] = (1, 0, float(p[1]), float(p[2]), float(p[3]))
        self._flips = flips
        self._after_flip_change()

    def save_flips(self, path: str | None = None) -> None:
        path = path or self.flip_path
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write("# invest_helical_F_3D flip list.\n")
            fh.write("#   tdrot,tilt,narot = resulting Dynamo ZXZ angles (cols 7-9); "
                     "position (cols 24-26) unchanged.\n")
            fh.write("#   tilt_flip = 1 if the polarity (perpendicular-dyad) flip was "
                     "applied, else 0.\n")
            fh.write("#   rot_flip  = 1 if the 180-deg-about-axis (C2) flip was "
                     "applied, else 0.\n")
            fh.write("# tag\ttdrot\ttilt\tnarot\ttilt_flip\trot_flip\n")
            for tag in sorted(self._flips):
                tilt, rot, a1, a2, a3 = self._flips[tag]
                fh.write(f"{tag}\t{a1:.4f}\t{a2:.4f}\t{a3:.4f}\t{tilt}\t{rot}\n")
        os.replace(tmp, path)

    def _after_change(self) -> None:
        if self.autosave:
            self.save()
        self.changed.emit()

    def _after_flip_change(self) -> None:
        if self.autosave:
            if self._flips:
                self.save_flips()
            elif os.path.exists(self.flip_path):
                os.remove(self.flip_path)        # don't leave a stale/empty list
        self.changed.emit()


def _iter(tags):
    if isinstance(tags, (int, float)):
        return [tags]
    return tags
