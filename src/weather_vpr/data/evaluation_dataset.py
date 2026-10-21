"""Evaluate with explicit gallery/query place IDs; never infer labels from filenames."""
from pathlib import Path
import csv
from PIL import Image
from torch.utils.data import Dataset
from weather_vpr.data.paired_dataset import resolve_file

class LabeledImages(Dataset):
    def __init__(self, csv_file: Path, image_size=(224,224)):
        self.path=Path(csv_file).expanduser().resolve(strict=True)
        self.image_size=image_size
        with self.path.open(newline='',encoding='utf8') as handle:
            reader=csv.DictReader(handle)
            if not {'path','place_id'}.issubset(reader.fieldnames or []):
                raise ValueError('CSV must have columns: path,place_id')
            self.rows=[]
            for n,r in enumerate(reader,2):
                if not r['path'] or not r['place_id']:
                    raise ValueError(f'CSV {self.path} line {n}: missing path/place_id')
                self.rows.append((resolve_file(r['path'], self.path),r['place_id']))
        if not self.rows:
            raise ValueError(f'Empty evaluation CSV: {self.path}')
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        from workflow.reliability_cache import preprocess_teacher_image
        file,label=self.rows[i]
        with Image.open(file) as image:
            t=preprocess_teacher_image(image,self.image_size)
        return t,label
