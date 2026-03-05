# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import typing as tp
import warnings

import numpy as np
import pandas as pd
import pydantic
import torch
from tqdm import tqdm

import data_utils as du
from data_utils.base import Frequency, TimedArray
from data_utils.events import Event, EventTypesHelper
from data_utils.infra import MapInfra
from data_utils.segments import Segment

logger = logging.getLogger(__name__)


class Fmri(pydantic.BaseModel):
    _event_types_helper: EventTypesHelper
    _missing_default: torch.Tensor | None = None
    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")
    name: tp.Literal["Fmri"] = "Fmri"
    infra: MapInfra = MapInfra()


    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)

        super().__init_subclass__()

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self._event_types_helper = EventTypesHelper("Fmri")

    def prepare(
        self, obj: pd.DataFrame | tp.Sequence[Event] | tp.Sequence[Segment]
    ) -> None:
        from data_utils import helpers

        events = helpers.extract_events(obj, types=self._event_types_helper)

        self._get_data(events)
        if events:

            self(
                events[0],
                start=events[0].start,
                duration=0.001,
                trigger=events[0].to_dict(),
            )

    def __call__(
        self,
        events: tp.Any,
        start: float,
        duration: float,
        trigger: float | dict[str, tp.Any] | None = None,
    ) -> torch.Tensor:
        _input_events = events

        from data_utils import helpers

        assert duration >= 0.0, f"{duration} must be >= 0."
        event_types = self._event_types_helper.classes
        name = self.__class__.__name__

        events = helpers.extract_events(events, types=self._event_types_helper)

        if not events and self._missing_default is not None:
            default = self._missing_default
            freq = Frequency(1/1.49)
            if freq:
                n_times = max(1, freq.to_ind(duration))
                reps = [1 for _ in range(default.ndim)] + [n_times]
                default = default.unsqueeze(-1).repeat(reps)
            return default

        
        events = events[:1]
        tarrays = list(
            self._get_timed_arrays(events=events, start=start, duration=duration)
        )

        time_info: dict[str, tp.Any] = {
            "start": start,
            "frequency": 1/1.49,
            "duration": duration,
        }
        out = TimedArray(aggregation="sum", **time_info)
        for ta in tarrays:
            out += ta
        tensor = torch.from_numpy(out.data)
        if not tensor.ndim:
            tensor = tensor.unsqueeze(0)

        if self._missing_default is None:

            shape = tuple(tensor.shape[: -1])
            self._missing_default = torch.zeros(*shape, dtype=tensor.dtype)
        return tensor

    def _exclude_from_cache_uid(self) -> list[str]:
        return [
            "offset",
        ]

    def _preprocess_event(self, event: du.events.Fmri) -> np.ndarray:
        rec = event.read()
        data = rec.get_fdata()

        import nilearn.signal

        data = data.T

        shape = data.shape
        data = nilearn.signal.clean(
            data.reshape(shape[0], -1),
            detrend=False,
            high_pass=None,
            t_r=1.49,
            standardize="zscore_sample",
        )
        data = data.reshape(shape).T
        return data.astype(np.float32)

    @infra.apply(
        item_uid=lambda e: str(e.filepath),
        exclude_from_cache_uid=_exclude_from_cache_uid,
        cache_type="NumpyMemmapArray",
    )
    def _get_data(self, events: tp.List[du.events.Fmri]) -> tp.Iterable[np.ndarray]:
        for event in tqdm(events, disable=len(events) < 2, desc="Computing fmri data"):
            yield self._preprocess_event(event)

    def _get_timed_arrays(
        self, events: list[du.events.Fmri], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        freq = events[0].frequency
        for event, data in zip(events, self._get_data(events)):
            yield TimedArray(
                data=data,
                frequency=freq,
                start=event.start - 4.47,
                duration=event.duration,
            )
