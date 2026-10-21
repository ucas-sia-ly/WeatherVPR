"""Cross-batch detached day descriptors for multi-positive weather InfoNCE."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from weather_vpr.data.place_sampler import normalize_place


class DayDescriptorQueue:
    """A fixed-size FIFO of detached, normalized day descriptors.

    Compute ``contrastive_loss`` before ``enqueue`` for each batch. Capacity
    zero gives batch-only InfoNCE without changing the training call site.
    Only day descriptors enter the queue; the weather queries retain gradients.
    """

    def __init__(self, capacity: int, *, storage_device: str | torch.device = "cpu") -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
            raise ValueError("capacity must be a nonnegative integer")
        self.capacity = capacity
        self.storage_device = torch.device(storage_device)
        self._descriptors: torch.Tensor | None = None
        self._places: list[tuple[str, str] | None] = [None] * capacity
        self._next = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._descriptors = None
        self._places = [None] * self.capacity
        self._next = 0
        self._size = 0

    def enqueue(self, day_desc: torch.Tensor,
                place_ids: Sequence[tuple[object, object]]) -> None:
        if day_desc.ndim != 2:
            raise ValueError("day_desc must have shape [batch, descriptor_dim]")
        places = tuple(normalize_place(place) for place in place_ids)
        if len(places) != day_desc.shape[0] or day_desc.shape[1] < 1:
            raise ValueError("day_desc and place_ids shapes disagree")
        if self.capacity == 0 or len(places) == 0:
            return
        with torch.no_grad():
            detached = F.normalize(day_desc.detach().float(), dim=1).to(self.storage_device)
            if not torch.isfinite(detached).all():
                raise ValueError("day_desc contains non-finite values")
            if self._descriptors is None:
                self._descriptors = torch.empty((self.capacity, detached.shape[1]),
                                                dtype=torch.float32, device=self.storage_device)
            elif self._descriptors.shape[1] != detached.shape[1]:
                raise ValueError("descriptor dimension changed after queue initialization")
            for desc, place in zip(detached, places):
                self._descriptors[self._next].copy_(desc)
                self._places[self._next] = place
                self._next = (self._next + 1) % self.capacity
                self._size = min(self._size + 1, self.capacity)

    def contrastive_loss(self, student_desc: torch.Tensor, day_desc: torch.Tensor,
                         place_ids: Sequence[tuple[object, object]],
                         temperature: float = 0.1) -> dict[str, torch.Tensor | int | float | list[int]]:
        """Contrast current weather queries against current and queued day keys.

        All keys with the same ``(city, place_id)`` are positives. The returned
        negative counts are descriptor counts, summed or listed per query, and
        can be written directly to JSON training logs.
        """
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if student_desc.ndim != 2 or day_desc.ndim != 2 or student_desc.shape != day_desc.shape:
            raise ValueError("student_desc and day_desc must have matching [batch, dim] shapes")
        if student_desc.shape[0] < 1 or student_desc.shape[1] < 1:
            raise ValueError("descriptor batch and dimension must be nonempty")
        places = tuple(normalize_place(place) for place in place_ids)
        if len(places) != student_desc.shape[0]:
            raise ValueError("place_ids must match batch length")
        if self._descriptors is not None and self._descriptors.shape[1] != student_desc.shape[1]:
            raise ValueError("descriptor dimension differs from queued day descriptors")

        query = F.normalize(student_desc.float(), dim=1)
        current_day = F.normalize(day_desc.detach().float().to(query.device), dim=1)
        keys = [current_day]
        key_places = list(places)
        if self._size:
            assert self._descriptors is not None
            keys.append(self._descriptors[:self._size].detach().to(query.device))
            queued_places = self._places[:self._size]
            if any(place is None for place in queued_places):
                raise RuntimeError("Queue place labels are incomplete")
            key_places.extend(place for place in queued_places if place is not None)
        all_keys = torch.cat(keys, dim=0)
        logits = (query @ all_keys.T) / temperature

        label_index = {place: index for index, place in enumerate(dict.fromkeys(key_places))}
        query_labels = torch.tensor([label_index[place] for place in places], device=query.device)
        key_labels = torch.tensor([label_index[place] for place in key_places], device=query.device)
        positives = query_labels[:, None] == key_labels[None, :]
        negative_counts = (~positives).sum(dim=1)
        log_positive = torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
        loss = (torch.logsumexp(logits, dim=1) - log_positive).mean()
        return {
            "loss": loss,
            "effective_negative_pairs": int(negative_counts.sum().item()),
            "effective_negatives_per_query": negative_counts.tolist(),
            "mean_effective_negatives": float(negative_counts.float().mean().item()),
            "positive_pairs": int(positives.sum().item()),
            "queue_size": self._size,
        }
