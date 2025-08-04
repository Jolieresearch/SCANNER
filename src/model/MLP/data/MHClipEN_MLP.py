import pandas as pd
import torch
from pathlib import Path
from PIL import Image
import os

from ...Base.data.MHClipEN_base import MHClipEN_Dataset


class MHClipEN_MLP_Dataset(MHClipEN_Dataset):
    def __init__(self, fold: int, split: str, task: str, **kargs):
        super(MHClipEN_MLP_Dataset, self).__init__()
        self.ocr_text = pd.read_json('data/MultiHateClip/en/ocr.jsonl', lines=True)
        self.data = self._get_data(fold, split, task)
        self.frame_path = Path('data/MultiHateClip/en/frames_16')
        self.title_text = pd.read_json('data/MultiHateClip/en/title.jsonl', lines=True)
        self.transcript_text = pd.read_json('data/MultiHateClip/en/speech.jsonl', lines=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        label = item['label']
        vid = item['Video_ID']

        cover_path = self.frame_path / vid / 'frame_000.jpg'
        if not os.path.exists(cover_path):
            cover_image = Image.new('RGB', (224, 224), color='black')
        else:
            cover_image = Image.open(cover_path).convert('RGB')

        title_text = self.title_text[self.title_text['vid'] == vid]['text'].values[0]
        transcript_text = self.transcript_text[self.transcript_text['vid'] == vid]['transcript'].values[0]
        ocr_text = self.ocr_text[self.ocr_text['vid'] == vid]['ocr'].values[0]

        new_text = ocr_text + ' ' + title_text
        
        return {
            'vid': vid,
            'cover_image': cover_image,
            # 'title_text': title_text,
            'transcript_text': transcript_text,
            'ocr_text': new_text,
            # 'all_text': new_text,
            'label': torch.tensor(label)
        }
        # return {
        #     'vid': vid,
        #     'cover_image': cover_image,
        #     # 'title_text': title_text,
        #     'transcript_text': transcript_text,
        #     # 'ocr_text': ocr_text,
        #     'ocr_text': new_text,
        #     'label': torch.tensor(label)
        # }


class MHClipEN_MLP_Collator:
    def __init__(self, **kargs):
        # self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        # self.processor = AutoProcessor.from_pretrained(encoder_name, trust_remote_code=True)
        pass

    def __call__(self, batch):
        vids = [item['vid'] for item in batch]
        images = [item['cover_image'] for item in batch]
        # title_texts = [item['title_text'] for item in batch]
        transcript_texts = [item['transcript_text'] for item in batch]
        ocr_texts = [item['ocr_text'] for item in batch]
        # texts = [item['all_text'] for item in batch]
        labels = [item['label'] for item in batch]
        
        return {
            'vids': vids,
            'images': images, 
            'transcript_texts': transcript_texts,
            'ocr_texts': ocr_texts, 
            'labels': torch.stack(labels)
        }
