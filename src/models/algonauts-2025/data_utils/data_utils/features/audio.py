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
from torch import nn
from torch.nn import functional as F

import data_utils as du
from data_utils.base import Frequency, TimedArray
from data_utils.events import Event, EventTypesHelper
from data_utils.infra import MapInfra
from data_utils.segments import Segment

logger = logging.getLogger(__name__)


class Wav2VecBert(pydantic.BaseModel):

    name: tp.Literal["Wav2VecBert"] = "Wav2VecBert"


    device: tp.Literal["auto", "cpu", "cuda"] = "auto"
    layer_aggregation: tp.Literal["group_mean"] | None = "group_mean"

    _model: nn.Module
    _feature_extractor: nn.Module

    infra: MapInfra = MapInfra()
    _event_types_helper: EventTypesHelper
    _missing_default: torch.Tensor | None = None
    layers: list[float] = [0.5, 0.75, 1.0]
    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")

    def _get_sound_model(self) -> torch.nn.Module:
        from transformers import Wav2Vec2BertModel

        _model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        _model.to(self.device)
        _model.eval()
        return _model


    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self._event_types_helper = EventTypesHelper("Sound")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

        events = helpers.extract_events(events, types=self._event_types_helper)

        if not events and self._missing_default is not None:
            default = self._missing_default
            freq = Frequency(2.0)
            if freq:
                n_times = max(1, freq.to_ind(duration))
                reps = [1 for _ in range(default.ndim)] + [n_times]
                default = default.unsqueeze(-1).repeat(reps)
            return default

        

        tarrays = list(
            self._get_timed_arrays(events=events, start=start, duration=duration)
        )

        time_info: dict[str, tp.Any] = {
            "start": start,
            "frequency": 2.0,
            "duration": duration,
        }
        out = TimedArray(aggregation="sum", **time_info)
        for ta in tarrays:
            out += ta
        tensor = torch.from_numpy(out.data)
        if not tensor.ndim:
            tensor = tensor.unsqueeze(0)

        if self._missing_default is None:

            shape = tuple(tensor.shape[:-1])
            self._missing_default = torch.zeros(*shape, dtype=tensor.dtype)
        return tensor


    def _preprocess_wav(self, wav: torch.Tensor) -> torch.Tensor:
        wav = torch.mean(wav, dim=1)

        wav = (wav - wav.mean()) / (1e-8 + wav.std())
        return wav

    def _resample_wav(
        self, wav: torch.Tensor, old_frequency: float, new_frequency: float
    ) -> torch.Tensor:
        old_frequency, new_frequency = int(old_frequency), int(new_frequency)
        import julius

        wav = julius.resample.ResampleFrac(old_sr=old_frequency, new_sr=new_frequency)(
            wav.T
        ).T
        return wav

    @infra.apply(
        item_uid=lambda event: f"{event.filepath}_{event.offset:.2f}_{event.duration:.2f}",
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MemmapArrayFile",
    )
    def _get_data(self, events: list[du.events.Event]) -> tp.Iterator[np.ndarray]:
        if len(events) > 1:
            from tqdm import tqdm

            events = tqdm(events, desc="Computing audio embeddings")

        for event in events:
            if isinstance(event, du.events.Sound):
                wav = event.read()
                sfreq = event.frequency
            elif isinstance(event, du.events.Video):
                audio = event.read().audio
                wav = torch.tensor(audio.to_soundarray(), dtype=torch.float32)
                sfreq = audio.fps
            wav = self._resample_wav(wav, sfreq, self._input_frequency)
            wav = self._preprocess_wav(wav)
            latents = self._process_wav(wav)

            timepoints = Frequency(2.0).to_ind(event.duration)

            if abs(timepoints - latents.shape[-1]) > 0:
                if len(latents.shape) == 2:

                    latents = F.interpolate(latents[None], timepoints)[0]
                else:

                    latents = F.interpolate(latents, timepoints)
            yield latents.numpy()

    def _aggregate_layers(self, latents: np.ndarray) -> np.ndarray:
        layer_indices = np.unique(
            [int(i * (latents.shape[0] - 1)) for i in self.layers]
        ).tolist()

        if len(layer_indices) == 1:
            if self.layer_aggregation is None:
                return latents[layer_indices[0]][None, :]
            else:
                return latents[layer_indices[0]]
        else:
            if self.layer_aggregation == "group_mean":
                groups = []
                layer_indices[-1] += 1
                for l1, l2 in zip(layer_indices[:-1], layer_indices[1:]):
                    groups.append(latents[l1:l2].mean(0))
                return np.stack(groups)
            elif self.layer_aggregation is None:
                return latents[layer_indices]
            else:
                raise ValueError(f"Unknown layer aggregation: {self.layer_aggregation}")

    @property
    def _input_frequency(self) -> float:
        return getattr(self.feature_extractor, "sampling_rate", 16_000)

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return ["device"]

    def _exclude_from_cache_uid(self) -> list[str]:
        return ["device"] + ["layers", "layer_aggregation"]

    @property
    def feature_extractor(self) -> nn.Module:
        if not hasattr(self, "_feature_extractor"):
            self._feature_extractor = self._get_feature_extractor()
        return self._feature_extractor

    @property
    def model(self) -> nn.Module:
        if not hasattr(self, "_model"):
            self._model = self._get_sound_model()
        return self._model

    def _get_feature_extractor(self) -> torch.nn.Module:
        from transformers import AutoFeatureExtractor

        return AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    def _get_features(self, wav):
        out = self._feature_extractor(
            wav,
            return_tensors="pt",
            sampling_rate=self.feature_extractor.sampling_rate,
            do_normalize=True,
        )
        try:
            return out["input_features"]
        except KeyError:
            return out["input_values"]

    def _get_timed_arrays(
        self, events: list[du.events.Event], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        freq = 2.0
        for latent, event in zip(self._get_data(events), events):
            if freq is None:

                freq = latent.shape[-1] / event.duration

            tdata = TimedArray(data=latent, start=event.start, frequency=freq)
            sub = tdata.overlap(start=start, duration=duration)
            if sub is None:

                sub = tdata.overlap(start=tdata.start, duration=0)
            sub.data = self._aggregate_layers(sub.data)
            yield sub

    def _process_wav(self, wav: torch.Tensor) -> torch.Tensor:
        features = self._get_features(wav)
        with torch.no_grad():
            outputs = self.model(features.to(self.device), output_hidden_states=True)
        out: tp.Any = outputs.get("hidden_states")
        if isinstance(out, tuple):
            out = torch.stack(out)

        out = out.squeeze(1).detach().cpu().clone().transpose(-1, -2).numpy()

        return torch.Tensor(out)
