# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

def configure_model(model):
    model.train()
    model.requires_grad_(False)

    modules_to_adapt = [
        model.linear_vision,
        model.linear_text,
        model.adapt_transformer
    ]

    for parent_module in modules_to_adapt:
        for module in parent_module.modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                module.requires_grad_(True)
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print(f"TTA - Total {len(trainable_params)} trainable parameters found (Encoder is frozen).")
    return model

class MLPWithClustering(nn.Module):
    def __init__(self, source_model, device, k=15, lambda_cluster=6, momentum=0.9, 
                 cluster_percentile=0.6, soft_threshold_alpha=5.0, lambda_diversity_intra=0.1, min_cluster_size=3):
        super().__init__()
        self.model = source_model
        self.k = k
        self.lambda_cluster = lambda_cluster
        self.device = device
        self.momentum = momentum

        if not (0.0 < cluster_percentile <= 1.0):
            raise ValueError("cluster_percentile must in (0, 1]")
        self.cluster_percentile = cluster_percentile
        self.alpha = soft_threshold_alpha

        self.lambda_diversity_intra = lambda_diversity_intra
        self.min_cluster_size = min_cluster_size

        self.centroids_image = None
        self.centroids_ocr = None
        self.centroids_trans = None

    def update_centroids_from_features(self, features, old_centroids):
        kmeans = KMeans(n_clusters=self.k, n_init='auto', random_state=0).fit(features.cpu().numpy())
        new_centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)
        if old_centroids is not None:
            return self.momentum * old_centroids + (1 - self.momentum) * new_centroids
        return new_centroids

    def compute_centroids(self, loader):
        original_mode = self.model.training
        try:
            self.model.eval()
            image_feats, ocr_feats, trans_feats = [], [], []
            with torch.no_grad():
                for batch in loader:
                    fea_image, fea_ocr, fea_trans = self.extract_modal_features(batch)
                    image_feats.append(fea_image.detach().cpu())
                    ocr_feats.append(fea_ocr.detach().cpu())
                    trans_feats.append(fea_trans.detach().cpu())

            image_feats = torch.cat(image_feats)
            ocr_feats = torch.cat(ocr_feats)
            trans_feats = torch.cat(trans_feats)

            self.centroids_image = self.update_centroids_from_features(image_feats, self.centroids_image)
            self.centroids_ocr = self.update_centroids_from_features(ocr_feats, self.centroids_ocr)
            self.centroids_trans = self.update_centroids_from_features(trans_feats, self.centroids_trans)
        finally:
            self.model.train(original_mode)

    def extract_modal_features(self, inputs):
        image_encoder = self.model.image_encoder
        text_encoder = self.model.text_encoder
        linear_vision = self.model.linear_vision
        linear_text = self.model.linear_text

        image_inputs = inputs['image_inputs'].to(self.device)
        transcript_texts_inputs = inputs['transcript_texts_inputs'].to(self.device)
        ocr_texts_inputs = inputs['ocr_texts_inputs'].to(self.device)

        fea_image = image_encoder(**image_inputs)['pooler_output']
        fea_trans = text_encoder(**transcript_texts_inputs)['pooler_output']
        fea_ocr = text_encoder(**ocr_texts_inputs)['pooler_output']

        fea_image = linear_vision(fea_image.to(self.device))
        fea_ocr = linear_text(fea_ocr.to(self.device))
        fea_trans = linear_text(fea_trans.to(self.device))

        return fea_image, fea_ocr, fea_trans

    def forward(self, inputs):
        out, fea = self.model(**inputs)

        if self.centroids_image is None:
            return out, torch.tensor(0.0, device=self.device)

        fea_image, fea_ocr, fea_trans = self.extract_modal_features(inputs)

        fea_image_norm = F.normalize(fea_image, dim=1)
        fea_ocr_norm = F.normalize(fea_ocr, dim=1)
        fea_trans_norm = F.normalize(fea_trans, dim=1)

        cent_img_norm = F.normalize(self.centroids_image, dim=1)
        cent_ocr_norm = F.normalize(self.centroids_ocr, dim=1)
        cent_trans_norm = F.normalize(self.centroids_trans, dim=1)

        sim_img = F.cosine_similarity(fea_image_norm.unsqueeze(1), cent_img_norm.unsqueeze(0), dim=2)
        sim_ocr = F.cosine_similarity(fea_ocr_norm.unsqueeze(1), cent_ocr_norm.unsqueeze(0), dim=2)
        sim_trans = F.cosine_similarity(fea_trans_norm.unsqueeze(1), cent_trans_norm.unsqueeze(0), dim=2)

        def _calculate_attractive_loss(avg_sim_scores):
            if avg_sim_scores.numel() == 0: return torch.tensor(0.0, device=self.device)
            num_samples_to_keep = int(avg_sim_scores.numel() * self.cluster_percentile)
            if num_samples_to_keep == 0: return torch.tensor(0.0, device=self.device)
            k_th_smallest_index = avg_sim_scores.numel() - num_samples_to_keep
            dynamic_threshold = torch.kthvalue(avg_sim_scores.detach(), k_th_smallest_index + 1)[0]
            weights = torch.sigmoid(self.alpha * (avg_sim_scores - dynamic_threshold))
            return (weights * (1 - avg_sim_scores)).mean()

        loss_attr_img = _calculate_attractive_loss(sim_img.mean(dim=1))
        loss_attr_ocr = _calculate_attractive_loss(sim_ocr.mean(dim=1))
        loss_attr_trans = _calculate_attractive_loss(sim_trans.mean(dim=1))
        total_attractive_loss = loss_attr_img + loss_attr_ocr + loss_attr_trans

        _, assignments_v = sim_img.max(dim=1)
        _, assignments_t = sim_trans.max(dim=1)
        _, assignments_a = sim_ocr.max(dim=1)

        def compute_diversity_loss(assignments, out):
            total_loss = torch.tensor(0.0, device=self.device)
            for i in range(self.k):
                cluster_mask = (assignments == i)
                if cluster_mask.sum() >= self.min_cluster_size:
                    cluster_logits = out[cluster_mask]
                    cluster_mean_probs = cluster_logits.softmax(dim=-1).mean(dim=0)
                    loss = (cluster_mean_probs * torch.log(cluster_mean_probs + 1e-8)).sum()
                    total_loss += loss
            return total_loss

        diversity_loss_v = compute_diversity_loss(assignments_v, out)
        diversity_loss_t = compute_diversity_loss(assignments_t, out)
        diversity_loss_a = compute_diversity_loss(assignments_a, out)
        total_diversity_loss = diversity_loss_v + diversity_loss_t + diversity_loss_a

        final_total_loss = (self.lambda_cluster * total_attractive_loss) + (self.lambda_diversity_intra * total_diversity_loss)

        return out, final_total_loss