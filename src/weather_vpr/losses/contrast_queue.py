"""Detached Day keys for multi-positive retrieval and relation distillation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from weather_vpr.data.place_sampler import normalize_place


def _fp32(tensor: torch.Tensor):
    if tensor.device.type in ("cpu", "cuda"):
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def _validate_temperature(value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be finite and positive")


def _validate_hard_negatives(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("hard_negatives must be a nonnegative integer")


def _select_negatives(similarities: torch.Tensor, candidates: torch.Tensor,
                      hard_negatives: int) -> torch.Tensor:
    """Select by detached similarities; zero keeps every eligible negative."""
    if hard_negatives == 0:
        return candidates
    scores = similarities.detach().masked_fill(~candidates, -torch.inf)
    indices = scores.topk(min(hard_negatives, scores.shape[1]), dim=1).indices
    selected = torch.zeros_like(candidates).scatter_(1, indices, True)
    return selected & candidates


class DayDescriptorQueue:
    """A fixed-size FIFO of detached, normalized day descriptors.

    Compute losses before ``enqueue`` for each batch. Capacity
    zero gives batch-only InfoNCE without changing the training call site.
    Only day descriptors enter the queue; the weather queries retain gradients.
    """

    def __init__(self, capacity: int, *, storage_device: str | torch.device = "cpu",
                 include_current_keys: bool = True) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
            raise ValueError("capacity must be a nonnegative integer")
        self.capacity = capacity
        if type(include_current_keys) is not bool:
            raise ValueError('include_current_keys must be boolean')
        self.include_current_keys = include_current_keys
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
        if day_desc.ndim != 2 or not day_desc.is_floating_point():
            raise ValueError("day_desc must have shape [batch, descriptor_dim]")
        places = tuple(normalize_place(place) for place in place_ids)
        if len(places) != day_desc.shape[0] or day_desc.shape[1] < 1:
            raise ValueError("day_desc and place_ids shapes disagree")
        if self.capacity == 0 or len(places) == 0:
            return
        with torch.no_grad(), _fp32(day_desc):
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

    def _inputs(self, student_desc: torch.Tensor, day_desc: torch.Tensor,
                place_ids: Sequence[tuple[object, object]]):
        """Validate queries and construct normalized, detached Day keys."""
        if student_desc.ndim != 2 or day_desc.ndim != 2 or student_desc.shape != day_desc.shape:
            raise ValueError("student_desc and day_desc must have matching [batch, dim] shapes")
        if not student_desc.is_floating_point() or not day_desc.is_floating_point():
            raise ValueError("descriptors must be floating point")
        if student_desc.shape[0] < 1 or student_desc.shape[1] < 1:
            raise ValueError("descriptor batch and dimension must be nonempty")
        if not torch.isfinite(student_desc).all() or not torch.isfinite(day_desc).all():
            raise ValueError("descriptors contain non-finite values")
        places = tuple(normalize_place(place) for place in place_ids)
        if len(places) != student_desc.shape[0]:
            raise ValueError("place_ids must match batch length")
        if self._descriptors is not None and self._descriptors.shape[1] != student_desc.shape[1]:
            raise ValueError("descriptor dimension differs from queued day descriptors")

        query = F.normalize(student_desc.float(), dim=1)
        current_day = F.normalize(day_desc.detach().float().to(query.device), dim=1)
        keys = [current_day] if self.include_current_keys else []
        key_places = list(places) if self.include_current_keys else []
        if self._size:
            assert self._descriptors is not None
            keys.append(self._descriptors[:self._size].detach().to(query.device))
            queued_places = self._places[:self._size]
            if any(place is None for place in queued_places):
                raise RuntimeError("Queue place labels are incomplete")
            key_places.extend(place for place in queued_places if place is not None)
        if not keys:
            raise ValueError('A fixed Day bank must be populated before computing losses')
        all_keys = torch.cat(keys, dim=0)
        label_index = {place: index for index, place in enumerate(dict.fromkeys(key_places))}
        if any(place not in label_index for place in places):
            raise ValueError('Every query place must have a positive in the Day bank')
        query_labels = torch.tensor([label_index[place] for place in places], device=query.device)
        key_labels = torch.tensor([label_index[place] for place in key_places], device=query.device)
        positives = query_labels[:, None] == key_labels[None, :]
        return query, current_day, all_keys, positives, key_labels

    def contrastive_loss(self, student_desc: torch.Tensor, day_desc: torch.Tensor,
                         place_ids: Sequence[tuple[object, object]],
                         temperature: float = 0.1,
                         hard_negatives: int = 0) -> dict[str, torch.Tensor | int | float | list[int]]:
        """Multi-positive InfoNCE, optionally using the Student's hardest K negatives.

        Every same-place key remains a positive. Candidate and selected counts
        count descriptors, not distinct places; the old effective counts refer
        to selected negatives. ``hard_negatives=0`` preserves batch/queue InfoNCE.
        Selection and all Day keys are detached, while selected Student logits
        retain their gradients. Calculations stay in FP32 under autocast.
        """
        _validate_temperature(temperature)
        _validate_hard_negatives(hard_negatives)
        with _fp32(student_desc):
            query, _, keys, positives, _ = self._inputs(student_desc, day_desc, place_ids)
            similarities = query @ keys.T
            candidates = ~positives
            selected = _select_negatives(similarities, candidates, hard_negatives)
            logits = similarities / temperature
            log_positive = torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
            log_all = torch.logsumexp(logits.masked_fill(~(positives | selected), -torch.inf), dim=1)
            loss = (log_all - log_positive).mean()
            candidate_counts = candidates.sum(dim=1)
            selected_counts = selected.sum(dim=1)
            return {
                "loss": loss,
                "effective_negative_pairs": int(selected_counts.sum().item()),
                "effective_negatives_per_query": selected_counts.tolist(),
                "mean_effective_negatives": float(selected_counts.float().mean().item()),
                "candidate_negative_pairs": int(candidate_counts.sum().item()),
                "candidate_negatives_per_query": candidate_counts.tolist(),
                "mean_candidate_negatives": float(candidate_counts.float().mean().item()),
                "selected_negative_pairs": int(selected_counts.sum().item()),
                "selected_negatives_per_query": selected_counts.tolist(),
                "mean_selected_negatives": float(selected_counts.float().mean().item()),
                "positive_pairs": int(positives.sum().item()),
                "queue_size": self._size,
            }

    def relational_distillation_loss(
            self, student_desc: torch.Tensor, day_desc: torch.Tensor,
            place_ids: Sequence[tuple[object, object]],
            student_temperature: float = 0.1, teacher_temperature: float = 0.1,
            hard_negatives: int = 64) -> dict[str, torch.Tensor | int | float | list[int]]:
        """Match Day-to-Day and Night-to-Day relation distributions with KL(T || S).

        Same-place keys, including queued positives, are excluded from both
        distributions. The detached Teacher chooses up to K remaining keys
        (zero uses all). A query contributes only when the selected support
        contains at least two keys from at least two different negative places.
        This is relation distillation inspired by GCISG, not a reproduction of it.
        """
        _validate_temperature(student_temperature)
        _validate_temperature(teacher_temperature)
        _validate_hard_negatives(hard_negatives)
        with _fp32(student_desc):
            query, teacher, keys, positives, key_labels = self._inputs(
                student_desc, day_desc, place_ids)
            teacher_similarities = teacher @ keys.T
            candidates = ~positives
            support = _select_negatives(teacher_similarities, candidates, hard_negatives)
            support_counts = support.sum(dim=1)
            # Counting labels is metadata only; no correspondence or Teacher
            # target receives a gradient through this validity decision.
            support_places = [int(torch.unique(key_labels[mask]).numel()) for mask in support]
            valid = (support_counts >= 2) & torch.tensor(
                [count >= 2 for count in support_places], device=query.device)
            valid_queries = int(valid.sum().item())
            loss = query.sum() * 0.0
            entropy = 0.0
            if valid_queries:
                valid_support = support[valid]
                teacher_logits = (teacher_similarities[valid] / teacher_temperature).masked_fill(
                    ~valid_support, -torch.inf)
                student_logits = ((query[valid] @ keys.T) / student_temperature).masked_fill(
                    ~valid_support, -torch.inf)
                teacher_log = F.log_softmax(teacher_logits, dim=1).masked_fill(~valid_support, 0)
                student_log = F.log_softmax(student_logits, dim=1).masked_fill(~valid_support, 0)
                teacher_probs = teacher_log.exp() * valid_support
                loss = (teacher_probs * (teacher_log - student_log)).sum(dim=1).mean()
                entropy = float(-(teacher_probs * teacher_log).sum(dim=1).mean().item())
            candidate_counts = candidates.sum(dim=1)
            return {
                "loss": loss,
                "relation_loss": loss,
                "valid_queries": valid_queries,
                "support_per_query": support_counts.tolist(),
                "mean_support": float(support_counts[valid].float().mean().item()) if valid_queries else 0.0,
                "support_places_per_query": support_places,
                "teacher_entropy": entropy,
                "candidate_negative_pairs": int(candidate_counts.sum().item()),
                "candidate_negatives_per_query": candidate_counts.tolist(),
                "mean_candidate_negatives": float(candidate_counts.float().mean().item()),
                "selected_negative_pairs": int(support_counts.sum().item()),
                "queue_size": self._size,
            }
