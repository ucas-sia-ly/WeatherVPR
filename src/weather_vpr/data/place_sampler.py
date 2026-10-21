"""Batch sampling that spreads GSV city/place IDs across each batch."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


def normalize_place(value: tuple[object, object]) -> tuple[str, str]:
    """Use the same composite label convention as AcceptedWeatherPairs."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"Place must be (city, place_id), got {value!r}")
    city, place_id = value
    if city is None or place_id is None or str(city) == "" or str(place_id) == "":
        raise ValueError(f"Invalid place: {value!r}")
    return str(city), str(place_id)


class DistinctPlaceBatchSampler(Sampler[list[int]]):
    """Use each pair once per epoch, with distinct places whenever possible.

    Pass ``[row['place'] for row in dataset.rows]`` as ``place_ids``. A place
    with multiple images may appear more than once in a batch only when fewer
    than ``batch_size`` places still have unused images. Call ``set_epoch`` at
    the start of each epoch to obtain a new deterministic order.
    """

    def __init__(self, place_ids: Sequence[tuple[object, object]], batch_size: int,
                 *, seed: int = 42, drop_last: bool = False) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        self.place_ids = tuple(normalize_place(place) for place in place_ids)
        if not self.place_ids:
            raise ValueError("place_ids must not be empty")
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        count = len(self.place_ids)
        return count // self.batch_size if self.drop_last else (count + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, place in enumerate(self.place_ids):
            groups[place].append(index)
        places = list(groups)
        rng.shuffle(places)
        for indices in groups.values():
            rng.shuffle(indices)
        active = deque(places)
        batch: list[int] = []
        while active:
            place = active.popleft()
            batch.append(groups[place].pop())
            if groups[place]:
                active.append(place)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch
