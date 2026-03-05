# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import collections
import dataclasses
import logging
import typing as tp
import warnings

import numpy as np
import pandas as pd
import tqdm

from .events import Event
from .utils import warn_once

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Segment:

    start: float
    duration: float
    _index: np.ndarray

    ns_events: tp.List[Event] = dataclasses.field(default_factory=list)
    _trigger: float | tp.Dict[str, tp.Any] | None = None

    @property
    def events(self) -> pd.DataFrame:

        if not self.ns_events:
            raise RuntimeError(f"ns_events was not populated in {self}")
        if len(self.ns_events) != len(self._index):
            msg = f"Cannot recreate events dataframe as some rows were not actual Event\n(on segment={self})"
            raise RuntimeError(msg)
        return pd.DataFrame(index=self._index, data=[e.to_dict() for e in self.ns_events])

    def subsegment(self, start: float, duration: float) -> "Segment":

        assert (
            start >= 0
        ), "Start is relative to the segment start and must be non-negative"
        new_start = self.start + start
        new_duration = duration
        new_index, new_ns_events = [], []
        for i, e in enumerate(self.ns_events):
            if e.start <= new_start + new_duration and e.start + e.duration >= new_start:
                new_index.append(self._index[i])
                new_ns_events.append(e)
        new_index = np.array(new_index)
        return Segment(
            start=new_start,
            duration=new_duration,
            _index=new_index,
            ns_events=new_ns_events,
            _trigger=self._trigger,
        )

    @property
    def event_list(self) -> list[Event]:
        raise RuntimeError(
            "segment.event_list is deprecated in favor of segment.ns_events"
        )

    @property
    def stop(self) -> float:
        return self.start + self.duration

    def _to_feature(self) -> dict[str, tp.Any]:

        return {
            "start": self.start,
            "duration": self.duration,
            "events": self.ns_events,
            "trigger": self._trigger,
        }


def _validate_event(event: pd.Series) -> dict[str, tp.Any]:

    event_type = event["type"]
    lower = {x.lower() for x in Event._CLASSES}
    if event_type in Event._CLASSES:
        event_class = Event._CLASSES[event_type]
        event_obj = event_class.from_dict(event).to_dict()

        event_dict = {**event, **event_obj}
    elif event_type in lower:
        raise ValueError(f"Legacy uncapitalized event {event}")
    else:
        warn_once(
            f'Unexpected type "{event["type"]}". Support for new event '
            "types can be added by creating new `Event` classes in "
            "`data_utils.events`."
        )
        event_dict = {**event}

    return event_dict


def validate_events(events: pd.DataFrame) -> pd.DataFrame:

    if events.empty:
        return events.copy()
    msg = 'events DataFrame must have a "type" column with strings'
    if "type" not in events.keys():
        raise ValueError(msg)
    types = events["type"].unique()
    if not all(isinstance(typ, str) for typ in types):
        raise ValueError(msg)

    df = pd.DataFrame(
        events.apply(_validate_event, axis=1).tolist(),
        index=events.index,
    )

    null = df.loc[df.duration <= 0, :]
    if not null.empty:
        types = null["type"].unique()
        msg = f"Found {len(null)} event(s) with null duration (types: {types})"
        warnings.warn(msg)

    dfs = []
    for _, sub in df.groupby(by="timeline", sort=False):
        dfs.append(
            sub.sort_values(
                by=["start", "duration"], ascending=[True, False], ignore_index=True
            )
        )
    important = ["type", "start", "duration", "timeline"]
    df = pd.concat(dfs, ignore_index=True)

    columns = important + [c for c in df.columns if c not in important]
    df = df.loc[:, columns]

    df = df.assign(stop=lambda x: x.start + x.duration)
    return df


def _prepare_strided_windows(
    start: float,
    stop: float,
    stride: float,
    duration: float,
    drop_incomplete: bool = True,
) -> tuple[np.ndarray, np.ndarray]:

    eps = 1e-8
    if drop_incomplete:
        stop -= duration
    starts = np.arange(start, stop + eps, stride)
    durations = np.full_like(starts, fill_value=duration)
    return starts, durations


def iter_segments(
    events: pd.DataFrame,
) -> tp.Iterator[Segment]:

    starts: tp.Any
    durations: tp.Any
    creators = SegmentCreator.from_obj(events)

    for creator in creators.values():
        starts, durations = _prepare_strided_windows(
            creator.starts.min() - 4.47,
            creator.stops.max() - 4.47,
            149.0,
            149.0,
            drop_incomplete=False,
        )
        for start_, duration_ in zip(starts, durations):
            seg = creator.select(start=start_, duration=duration_)
            seg._trigger = start_
            yield seg
    return


def list_segments(
    events: pd.DataFrame,
) -> list[Segment]:
    return list(iter_segments(**locals()))


def find_enclosed(df: pd.DataFrame, start: float, duration: float) -> pd.Series:
    estart = np.array(df.start)
    estop = estart + np.array(df.duration)
    is_enclosed = np.logical_and(estart >= start, estop <= start + duration)
    return pd.Series(df.index[is_enclosed])


def find_overlap(
    events: pd.DataFrame,
    idx: int | pd.Series | None = None,
    *,
    start: float = 0.0,
    duration: float | np.ndarray | None = None,
) -> pd.Series:

    if idx is None:

        assert duration is not None
        assert events.timeline.nunique() == 1
        has_overlap = (events.start >= start) & (events.start < start + duration)
        has_overlap |= (events.start + events.duration > start) & (
            events.start + events.duration <= start + duration
        )
        has_overlap |= (events.start <= start) & (
            events.start + events.duration >= start + duration
        )

        out = events.index[has_overlap]
        return pd.Series(out)
    else:
        sel: list[int] = []
        for segment in iter_segments(
            events,
            idx=idx,
            start=start,
            duration=duration,
            stride=None,
        ):
            sel.extend(segment._index.tolist())

        return pd.Series(sel)


class SegmentCreator:

    def __init__(self, events: list[Event]) -> None:
        timelines = {e.timeline for e in events}
        if len(timelines) > 1:
            name = self.__class__.__name__
            msg = f"Cannot create {name} on several timelines, got {timelines}"
            raise ValueError(msg)
        self.events = np.array(events)
        self.starts = np.array([e.start for e in events])
        self.indices = np.array([e._index for e in events])
        self.stops = np.array([e.duration for e in events]) + self.starts

    @classmethod
    def from_obj(cls, obj: tp.Any) -> dict[str, "SegmentCreator"]:

        from data_utils import helpers

        timeline_events: dict[str, list[Event]] = collections.defaultdict(list)
        for e in helpers.extract_events(obj):
            timeline_events[e.timeline].append(e)
        timelines = list(timeline_events)
        if isinstance(obj, pd.DataFrame):

            timelines = list(obj.timeline.unique())
        return {tl: cls(timeline_events[tl]) for tl in timelines}

    def select(self, start: float, duration: float) -> Segment:

        select = self.starts < start + duration
        select &= self.stops > start
        events = list(self.events[select])
        index = self.indices[select]
        return Segment(ns_events=events, start=start, duration=duration, _index=index)
