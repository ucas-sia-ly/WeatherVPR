"""CSV-verified training Day images for an independent Source Night stream.

Labels come from the original GSV-Cities CSV rows. Filenames are checked
against those rows as provenance, never used to infer a place label. A signed
manifest sidecar records the complete exclusion union and its evidence.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from weather_vpr.data.paired_dataset import resolve_file, sha256
from weather_vpr.data.place_sampler import normalize_place
from weather_vpr.data.source_night import SourceNightAugmentation, SourceNightConfig
from weather_vpr.data.student_augmentation import sample_augmentation_seed, student_training_tensor
from weather_vpr.utils.backend import enable_backend_imports


CSV_FIELDS = ('place_id', 'year', 'month', 'northdeg', 'city_id', 'lat', 'lon', 'panoid')


def _integer(value, name):
    text = str(value)
    try:
        number = int(text)
    except (ValueError, TypeError) as error:
        raise ValueError(f'GSV {name} must be a CSV integer: {value!r}') from error
    # Preserve raw signed headings exactly, as the original SALAD dataset
    # does. Local CSV headings range from -11 to 731; wrapping/clamping them
    # would change the annotation-backed filenames.
    if (number < 0 and name != 'northdeg') or str(number) != text:
        raise ValueError(f'GSV {name} has an invalid canonical integer: {value!r}')
    return text


def gsv_image_name(row):
    """Original SALAD/GSV-Cities filename construction using a raw CSV row."""
    return (f"{row['city_id']}_{_integer(row['place_id'], 'place_id').zfill(7)}_"
            f"{_integer(row['year'], 'year').zfill(4)}_"
            f"{_integer(row['month'], 'month').zfill(2)}_"
            f"{_integer(row['northdeg'], 'northdeg').zfill(3)}_"
            f"{row['lat']}_{row['lon']}_{row['panoid']}.jpg")


def _identity(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _signed_file_map(value, manifest, label, *, required=False):
    if not isinstance(value, dict) or (required and not value):
        raise ValueError(f'GSV sidecar requires a nonempty {label} mapping')
    resolved = {}
    for name, expected in value.items():
        path = resolve_file(name, manifest)
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in '0123456789abcdef' for c in expected)):
            raise ValueError(f'Invalid {label} SHA-256 for {name}')
        if sha256(path) != expected:
            raise ValueError(f'GSV {label} file changed: {path}')
        if path in resolved:
            raise ValueError(f'Duplicate resolved {label} file: {path}')
        resolved[path] = expected
    return resolved


class GSVDayImages(Dataset):
    """A train-only Day pool whose composite labels have original CSV evidence."""

    def __init__(self, manifest, image_size=(224, 224), *, exclude_places=None,
                 source_night_config=None, seed=42):
        self.manifest = Path(manifest).expanduser().resolve(strict=True)
        self.size = tuple(image_size)
        if (len(self.size) != 2 or any(type(v) is not int or v < 1 for v in self.size)):
            raise ValueError('GSV image_size must contain two positive integers')
        self.seed, self.epoch = int(seed), 0
        self.source_night_config = source_night_config or SourceNightConfig()
        self.source_night_augmentation = SourceNightAugmentation(self.source_night_config)
        self.sidecar = Path(str(self.manifest) + '.metadata.json')
        sidecar_hash = sha256(self.sidecar)
        info = json.loads(self.sidecar.read_text(encoding='utf-8'))
        if (not isinstance(info, dict) or info.get('schema_version') != 1
                or info.get('kind') != 'gsv_day_train'):
            raise ValueError('GSV Day sidecar must be schema_version=1, kind=gsv_day_train')
        if sha256(self.manifest) != info.get('manifest_sha256'):
            raise ValueError('GSV Day manifest differs from its signed sidecar')
        csv_files = _signed_file_map(info.get('source_csvs'), self.manifest,
                                     'source_csvs', required=True)
        exclusion_files = _signed_file_map(info.get('exclusion_files'), self.manifest,
                                          'exclusion_files', required=True)
        exclusion_evidence = _signed_file_map(info.get('exclusion_evidence', {}),
                                             self.manifest, 'exclusion_evidence')
        self.excluded_places = set()
        for path in exclusion_files:
            rows = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(rows, list):
                raise ValueError(f'GSV exclusion file must be a place tuple list: {path}')
            self.excluded_places.update(normalize_place(row) for row in rows)
        if exclude_places is not None:
            path = Path(exclude_places).expanduser().resolve(strict=True)
            if path not in exclusion_files and path not in exclusion_evidence:
                raise ValueError('Configured --exclude-places is not signed by the GSV Day sidecar')
            rows = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(rows, list):
                raise ValueError('Configured --exclude-places must be a place tuple list')
            self.excluded_places.update(normalize_place(row) for row in rows)
        raw_rows = []
        requested = {path: set() for path in csv_files}
        for line_index, line in enumerate(self.manifest.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (not isinstance(row, dict) or row.get('schema_version') != 1
                    or row.get('kind') != 'gsv_day_train' or row.get('source') != 'gsv_day'
                    or row.get('split') != 'train'):
                raise ValueError(f'GSV row {line_index} must be verified gsv_day/train schema 1')
            metadata = row.get('metadata', {})
            if (metadata.get('grouping_kind') != 'verified_gsv_place'
                    or metadata.get('place_label_verified') is not True):
                raise ValueError(f'GSV row {line_index} lacks verified place provenance')
            csv_path = resolve_file(metadata['annotation_csv'], self.manifest)
            csv_index = metadata.get('annotation_row')
            if type(csv_index) is not int or csv_index < 1:
                raise ValueError('GSV annotation_row must be a positive 1-based CSV data row')
            if (csv_path not in csv_files
                    or metadata.get('annotation_sha256') != csv_files[csv_path]):
                raise ValueError(f'GSV row {line_index} annotation is not signed by the sidecar')
            requested[csv_path].add(csv_index)
            raw_rows.append((row, csv_path, csv_index))
        if not raw_rows:
            raise ValueError('GSV Day training manifest is empty')
        bindings = {}
        for path, indices in requested.items():
            with path.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                    raise ValueError(f'Unexpected original GSV CSV schema: {path}')
                for index, row in enumerate(reader, 1):
                    if index in indices:
                        bindings[path, index] = row
            if any((path, index) not in bindings for index in indices):
                raise ValueError(f'GSV annotation row is outside its original CSV: {path}')
        self.rows, self.day_sources = [], []
        ids, image_paths = set(), set()
        for row, csv_path, csv_index in raw_rows:
            binding = bindings[csv_path, csv_index]
            supplied = row['metadata'].get('csv_binding')
            if (not isinstance(supplied, dict) or set(supplied) != set(CSV_FIELDS)
                    or any(str(supplied[key]) != binding[key] for key in CSV_FIELDS)):
                raise ValueError(f'GSV csv_binding does not match original row {csv_index}: {csv_path}')
            place = normalize_place((binding['city_id'], _integer(binding['place_id'], 'place_id')))
            if normalize_place((row.get('city'), row.get('place_id'))) != place:
                raise ValueError('GSV place label does not match the original CSV')
            if row.get('group_id') != f'gsv:{place[0]}:{place[1]}':
                raise ValueError('GSV group_id does not match its verified composite place')
            if place in self.excluded_places:
                raise ValueError(f'GSV held-out place leaked into training: {place}')
            path = resolve_file(row['path'], self.manifest)
            expected_path = csv_path.parent.parent / 'Images' / place[0] / gsv_image_name(binding)
            if path != expected_path.resolve(strict=True):
                raise ValueError('GSV image path does not match its original CSV-backed image')
            sample_id = row.get('sample_id')
            if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
                raise ValueError('GSV sample IDs must be nonempty and unique')
            if path in image_paths:
                raise ValueError(f'Duplicate GSV Day image path: {path}')
            image_hash = sha256(path)
            if row.get('sha256') is not None and row['sha256'] != image_hash:
                raise ValueError(f'GSV Day image hash changed: {path}')
            ids.add(sample_id)
            image_paths.add(path)
            self.rows.append({**row, 'source_file': path, 'place': place})
            self.day_sources.append({'path': str(path), 'sha256': image_hash})
        self._signed_hashes = {self.manifest: info['manifest_sha256'],
                               self.sidecar: sidecar_hash, **csv_files,
                               **exclusion_files, **exclusion_evidence}
        for path, expected in self._signed_hashes.items():
            if sha256(path) != expected:
                raise RuntimeError(f'GSV Day provenance changed during validation: {path}')
        self._versions = {path: _identity(path) for path in self._signed_hashes}
        self._image_versions = {_row['source_file']: _identity(_row['source_file'])
                                for _row in self.rows}

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def assert_current(self):
        for path, version in self._versions.items():
            if _identity(path) != version:
                raise RuntimeError(f'GSV Day training provenance changed: {path}; restart')

    def sources_signature(self):
        self.assert_current()
        for path, version in self._image_versions.items():
            if _identity(path) != version:
                raise RuntimeError(f'GSV Day training image changed: {path}; restart')
        return {'purpose': 'verified_independent_gsv_day_source_stream_v1',
                'signed_files': {str(path): value for path, value in self._signed_hashes.items()},
                'source_images': self.day_sources,
                'places': [list(row['place']) for row in self.rows]}

    def _read_day(self, index):
        self.assert_current()
        source = self.day_sources[index]
        path = self.rows[index]['source_file']
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != source['sha256']:
            raise RuntimeError(f'GSV Day image changed after Teacher extraction: {path}; restart')
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            return image.convert('RGB')

    def source_view(self, index, *, seed):
        enable_backend_imports()
        return student_training_tensor(self._read_day(index), self.size,
                                       self.source_night_augmentation, seed=seed)

    def __getitem__(self, index):
        enable_backend_imports()
        from workflow.reliability_cache import preprocess_teacher_image
        day = self._read_day(index)
        source, record = student_training_tensor(
            day, self.size, self.source_night_augmentation,
            seed=sample_augmentation_seed(self.seed, self.epoch, index, view='source_night'))
        return {'index': index, 'day_tensor': preprocess_teacher_image(day, self.size),
                'source_night_tensor': source, 'source_night_record': record,
                'sample_id': self.rows[index]['sample_id'], 'place': self.rows[index]['place']}
