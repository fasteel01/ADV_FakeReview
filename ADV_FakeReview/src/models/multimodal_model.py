"""
실험 A "full" 모드의 실제 모델: BERT(text) + ResNet-50(image) + CLIP(text-image consistency)
+ Self-MM(auxiliary head) + MISA-lite(shared/specific fusion)

이 파일은 huggingface.co / torchvision 사전학습 가중치 다운로드가 필요해서
클라우드 샌드박스에서는 실행할 수 없다 (README 참고). 학교 서버나 Colab처럼
인터넷이 열려있는 GPU 환경에서 실행할 것.

설계:
  1) extract_features()로 BERT/ResNet/CLIP embedding을 한 번만 뽑아서 캐싱한다
     (사전학습 backbone은 freeze, feature extractor로만 사용 - 9일짜리 프로젝트에서
     backbone까지 fine-tuning하는 건 시간/오버피팅 리스크 대비 실익이 적다고 판단).
  2) 캐싱된 feature 위에서 fusion head(MISA-lite + Self-MM)만 학습한다.
     -> 한 번 캐싱해두면 fusion 구조 여러 개를 몇 초 단위로 빠르게 비교 가능.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from models.fusion_modules import BranchEncoder, SelfMMAuxHead, MISALiteFusion

BERT_MODEL_NAME = "bert-base-uncased"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
COMBINED_DIM = 128


class ReviewImageDataset(Dataset):
    """csv(id, text, label, ...) + data/raw/images/{id}.jpg 를 읽는 Dataset.
    이미지가 없는 샘플은 회색 placeholder로 대체(개수 세서 로그로 알려줌)."""

    def __init__(self, csv_path, image_dir):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.missing = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['ID']}.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            self.missing += 1
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        return {"id": row["ID"], "text": row["text"], "label": int(row["label"]), "image": image}


@torch.no_grad()
def extract_features(csv_path, image_dir, out_path, device="cuda" if torch.cuda.is_available() else "cpu",
                      batch_size=32):
    """BERT [CLS] embedding(768d) + ResNet-50 penultimate embedding(2048d)
    + CLIP text-image cosine similarity(1d)를 뽑아서 .npz로 캐싱"""
    from transformers import AutoTokenizer, AutoModel, CLIPModel, CLIPProcessor
    from torchvision import models as tv_models
    from torchvision.transforms import v2 as T

    print(f"[extract_features] {csv_path} -> {out_path} (device={device})")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device).eval()

    resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = nn.Identity()  # penultimate 2048-d feature
    resnet = resnet.to(device).eval()
    img_transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    dataset = ReviewImageDataset(csv_path, image_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=lambda batch: batch)

    all_bert, all_resnet, all_clip_sim, all_labels, all_ids = [], [], [], [], []

    for batch in loader:
        texts = [b["text"] for b in batch]
        images = [b["image"] for b in batch]
        labels = [b["label"] for b in batch]
        ids = [b["id"] for b in batch]

        # BERT [CLS]
        tok = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        bert_out = bert(**tok).last_hidden_state[:, 0, :]  # [CLS]
        all_bert.append(bert_out.cpu().numpy())

        # ResNet-50
        img_tensor = torch.stack([img_transform(im) for im in images]).to(device)
        resnet_out = resnet(img_tensor)
        all_resnet.append(resnet_out.cpu().numpy())

        # CLIP text-image cosine similarity (핵심: MMIM을 대신하는 cross-modal consistency feature)
        clip_inputs = clip_processor(text=texts, images=images, return_tensors="pt",
                                      padding=True, truncation=True).to(device)
        clip_out = clip_model(**clip_inputs)
        text_emb = F.normalize(clip_out.text_embeds, dim=-1)
        image_emb = F.normalize(clip_out.image_embeds, dim=-1)
        sim = (text_emb * image_emb).sum(dim=-1)  # cosine similarity per pair
        all_clip_sim.append(sim.cpu().numpy())

        all_labels.extend(labels)
        all_ids.extend(ids)

    np.savez(
        out_path,
        bert=np.concatenate(all_bert), resnet=np.concatenate(all_resnet),
        clip_sim=np.concatenate(all_clip_sim), label=np.array(all_labels), id=np.array(all_ids),
    )
    print(f"  missing images: {dataset.missing}/{len(dataset)}")
    print(f"  saved: {out_path}")


class MultimodalFusionModel(nn.Module):
    """BERT branch + ResNet branch -> (Self-MM aux + MISA-lite fusion) -> + CLIP consistency -> classifier"""

    def __init__(self, bert_dim=768, resnet_dim=2048, use_aux=True, use_misa=True, use_clip_consistency=True):
        super().__init__()
        self.use_aux = use_aux
        self.use_misa = use_misa
        self.use_clip_consistency = use_clip_consistency

        self.text_branch = BranchEncoder(bert_dim, COMBINED_DIM)
        self.image_branch = BranchEncoder(resnet_dim, COMBINED_DIM)

        if use_misa:
            self.fusion = MISALiteFusion(COMBINED_DIM, shared_dim=64, specific_dim=64, num_branches=2)
            fused_dim = 64 * 2 * 2
        else:
            fused_dim = COMBINED_DIM * 2

        if use_aux:
            self.aux_text = SelfMMAuxHead(COMBINED_DIM)
            self.aux_image = SelfMMAuxHead(COMBINED_DIM)

        clf_in_dim = fused_dim + (1 if use_clip_consistency else 0)
        self.classifier = nn.Linear(clf_in_dim, 2)

    def forward(self, bert_emb, resnet_emb, clip_sim):
        text_repr = self.text_branch(bert_emb)
        image_repr = self.image_branch(resnet_emb)

        ortho_loss = torch.tensor(0.0, device=bert_emb.device)
        if self.use_misa:
            fused, ortho_loss = self.fusion([text_repr, image_repr])
        else:
            fused = torch.cat([text_repr, image_repr], dim=-1)

        if self.use_clip_consistency:
            fused = torch.cat([fused, clip_sim.unsqueeze(-1)], dim=-1)

        main_logits = self.classifier(fused)

        aux_logits = None
        if self.use_aux:
            aux_logits = (self.aux_text(text_repr), self.aux_image(image_repr))

        return main_logits, aux_logits, ortho_loss
