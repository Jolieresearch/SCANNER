import torch
import torch.nn as nn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from model.SCANNER.modules.transformer import TransformerEncoder
from sentence_transformers import SentenceTransformer

class SCANNER(nn.Module):
    def __init__(self, text_encoder: str, image_encoder: str, dataset: str, fea_dim=128, dropout=0.1, task='binary', **kargs):
        super(SCANNER, self).__init__()
        
        self.text_encoder = SentenceTransformer(text_encoder).requires_grad_(False)
        self.image_encoder = SentenceTransformer(image_encoder).requires_grad_(False)
        
        num_classes = 2 if task == 'binary' else 3
        self.dataset = dataset

        self.linear_vision = nn.Sequential(
            nn.LazyLinear(fea_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fea_dim * 2, fea_dim),
            nn.BatchNorm1d(fea_dim)
        )

        self.linear_text = nn.Sequential(
            nn.LazyLinear(fea_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fea_dim * 2, fea_dim),
            nn.BatchNorm1d(fea_dim)
        )

        self.adapt_transformer = TransformerEncoder(
            embed_dim=fea_dim * 3,
            num_heads=8,
            layers=2,
            attn_dropout=0.1,
            res_dropout=0.1,
            relu_dropout=0.1,
            embed_dropout=0.25,
        )

        self.classifier = nn.Sequential(
            nn.LazyLinear(200), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(200, num_classes)
        )

    def forward(self, **inputs):
        
        fea_image = self.image_encoder.encode(inputs['images'], convert_to_tensor=True, show_progress_bar=False)
        fea_ocr_texts = self.text_encoder.encode(inputs['ocr_texts'], convert_to_tensor=True, show_progress_bar=False)
        fea_transcript_texts = self.text_encoder.encode(inputs['transcript_texts'], convert_to_tensor=True, show_progress_bar=False)

        fea_image = self.linear_vision(fea_image)
        fea_transcript_texts = self.linear_text(fea_transcript_texts)
        fea_ocr_texts = self.linear_text(fea_ocr_texts)
        
        fea = torch.cat([fea_image, fea_transcript_texts, fea_ocr_texts], dim=-1)

        fea = self.adapt_transformer(fea.unsqueeze(1)).squeeze(1)

        output = self.classifier(fea)
    
        return output, fea
