import pandas as pd
import torch
from pathlib import Path
from PIL import Image
import os

from ...Base.data.HateMM_base import HateMM_Dataset


class HateMM_MLP_Dataset(HateMM_Dataset):
    def __init__(self, fold: int, split: str, task: str, **kargs):
        super(HateMM_MLP_Dataset, self).__init__()
        self.ocr_text = pd.read_json('data/HateMM/ocr.jsonl', lines=True)
        self.data = self._get_data(fold, split, task)
        self.frame_path = Path('data/HateMM/frames_16')
        # self.title_text = pd.read_json('data/HateMM/en/title.jsonl', lines=True)
        self.transcript_text = pd.read_json('data/HateMM/speech.jsonl', lines=True)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        label = item['label']
        vid = item['Video_ID']

        cover_path = self.frame_path / vid / 'frame_000.jpg'
        # print(cover_path)
        if not os.path.exists(cover_path):
            cover_image = Image.new('RGB', (224, 224), color='black')
        else:
            cover_image = Image.open(cover_path).convert('RGB')
        
        # title_text = self.title_text[self.title_text['vid'] == vid]['text'].values[0]
        transcript_text = self.transcript_text[self.transcript_text['vid'] == vid]['transcript'].values[0]
        # text = transcript_text
        ocr_text = self.ocr_text[self.ocr_text['vid'] == vid]['ocr'].values[0]
        # new_text = transcript_text + '' + ocr_text
        
        return {
            'vid': vid,
            'cover_image': cover_image,
            'transcript_text': transcript_text,
            'ocr_text': ocr_text,
            'label': torch.tensor(label)
        }


class HateMM_MLP_Collator:
    def __init__(self, **kargs):
        pass

    def __call__(self, batch):
        vids = [item['vid'] for item in batch]
        images = [item['cover_image'] for item in batch]
        transcript_texts = [item['transcript_text'] for item in batch]
        ocr_texts = [item['ocr_text'] for item in batch]
        labels = [item['label'] for item in batch]
        
        return {
            'vids': vids,
            'images': images, 
            'transcript_texts': transcript_texts,
            'ocr_texts': ocr_texts, 
            'labels': torch.stack(labels)
        }
