
import math
import os
import shutil
import wandb
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
from datetime import datetime
import sys
import torch.nn as nn
import hydra
from datetime import datetime
import pandas as pd
import torch
import colorama
from colorama import Fore, Style
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from model.MLP.model.SCANNER import MLPWithClustering, configure_model

from utils.core_utils import (
    get_collator,
    get_dataset,
    get_optimizer,
    get_scheduler,
    load_model,
    set_seed,
    set_worker_seed,
    BinaryClassificationMetric,
    TernaryClassificationMetric,
    softmax_entropy
)


log_path = Path(f'log/EN2MM/{datetime.now().strftime("%m%d-%H%M%S%f")}')

try:
    source_script_path = Path(__file__).resolve()
except NameError:
    source_script_path = Path('interactive_session.py')

log_path.mkdir(parents=True, exist_ok=True)
destination_script_path = log_path / source_script_path.name

try:
    shutil.copy2(source_script_path, destination_script_path)
    print(f"The script was successfully backed up to: {destination_script_path}")
except Exception as e:
    print(f"Error while backing up script: {e}")

class Trainer():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = cfg.device

        self.task = cfg.task
        if cfg.task == 'binary':
            self.evaluator = BinaryClassificationMetric(self.device)
        elif cfg.task == 'ternary':
            self.evaluator = TernaryClassificationMetric(self.device)
        else:
            raise ValueError('task not supported')
        
        self.type = cfg.type
        self.model_name = cfg.model
        self.dataset_name = cfg.dataset
        self.batch_size = cfg.batch_size
        self.num_epoch = cfg.num_epoch
        self.lambda_pre = cfg.lambda_pre
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.save_path = log_path
        
        if cfg.type == '5-fold':
            raise ValueError('experiment type not supported')
            self.dataset_range = [2, 1, 3, 4, 5]
        elif cfg.type == 'default':
            self.dataset_range = ['default']
        else:
            raise ValueError('experiment type not supported')
        
        self.collator = get_collator(cfg.model, cfg.dataset, **cfg.encoder)

        self.model = load_model(cfg.model, cfg.dataset, **dict(cfg.encoder))
        state_dict = torch.load(cfg.pretrained_model)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        
        self.model = configure_model(self.model)
        trainables = [p for p in self.model.parameters() if p.requires_grad]
        print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

        self.wrapper = MLPWithClustering(
            source_model=self.model,
            device=cfg.device,
            k=cfg.k, 
            lambda_cluster=cfg.lambda_cluster,
            momentum=0.9, 
            cluster_percentile=cfg.cluster_percentile, 
            soft_threshold_alpha=cfg.soft_threshold_alpha,
            lambda_diversity_intra=cfg.lambda_diversity_intra,
        )
        self.wrapper.to(self.device)
        self.entropy_threshold = cfg.entropy_threshold
        self.adapt_steps = cfg.adapt_steps
    
    def _reset(self, cfg, fold, type):
        train_dataset = get_dataset(cfg.model, cfg.dataset, task=cfg.task, fold=fold, split='all')
        if hasattr(cfg, 'test') and cfg.test:
            logger.info(f"Using {cfg.test.dataset} as test dataset!")
            test_dataset = get_dataset(cfg.model, cfg.test.dataset, fold=fold, split='all', task=cfg.task)
        else:
            test_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='all', task=cfg.task)
        valid_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='all', task=cfg.task)

        logger.info(f"Train on: {cfg.dataset}, Valid on: {cfg.dataset}, Test on: {cfg.test.dataset if hasattr(cfg, 'test') else cfg.dataset}")

        self.train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(24, cfg.batch_size//2), shuffle=True, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
        self.test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(24, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(24, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
                
        steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)
        self.optimizer = get_optimizer(self.model, **dict(cfg.opt))
        self.scheduler = get_scheduler(self.optimizer, steps_per_epoch=steps_per_epoch, **dict(cfg.sche))
        
    def run(self):          

            for fold in self.dataset_range:
                self._reset(self.cfg, fold, self.type)
                logger.info(f"--- Starting Fold: {fold} ---")

                for epoch in range(self.cfg.num_epoch):
                    self._train(epoch=epoch)
                logger.info(f"{Fore.GREEN}All {self.num_epoch} epochs for fold {fold} are complete.{Style.RESET_ALL}")

    def _train(self, epoch: int):
        all_preds, all_labels = [], []

        pbar = tqdm(self.train_dataloader, desc=f"{Fore.BLUE}Train Epoch {epoch}")

        if epoch == 0:
            trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
            logger.info(f"Trainable layers:\n" + "\n".join(trainable))
            logger.info("Initial updating clustering centroids...")
            self.wrapper.compute_centroids(self.train_dataloader)

        for step, batch in enumerate(pbar):
            _ = batch.pop('vids')
            labels = batch.pop('labels').to(self.device)

            if step == 2:
                logger.info("=== Switching to Stage 2 fine-tuning at batch 2 ===")
                logger.info("Unfreezing linear layers for Stage 2...")

                for module in self.model.linear_vision.modules():
                    if isinstance(module, nn.Linear):
                        for param in module.parameters():
                            param.requires_grad = True
                for module in self.model.linear_text.modules():
                    if isinstance(module, nn.Linear):
                        for param in module.parameters():
                            param.requires_grad = True

                trainable_params = [name for name, param in self.model.named_parameters() if param.requires_grad]
                logger.info(f"Stage 2: Found {len(trainable_params)} trainable parameters.")
                for name in trainable_params:
                    logger.debug(f"  - Trainable: {name}")

                logger.info("Rebuilding optimizer and scheduler for Stage 2...")
                stage2_lr = self.cfg.opt.lr * 0.1
                optimizer_cfg = dict(self.cfg.opt)
                optimizer_cfg['lr'] = stage2_lr
                self.optimizer = get_optimizer(self.model, **optimizer_cfg)
                self.scheduler = get_scheduler(self.optimizer, steps_per_epoch=len(self.train_dataloader), **dict(self.cfg.sche))

            inner_steps = self.adapt_steps if step >= 2 else 1

            for inner_iter in range(inner_steps):
                logits, cluster_loss = self.wrapper(batch)
                entropys = softmax_entropy(logits)

                current_threshold = self.entropy_threshold
                reliable_ids = torch.where(entropys < current_threshold)[0]

                if reliable_ids.numel() > 0:
                    selected_outputs = logits[reliable_ids]
                    selected_entropy = -(selected_outputs.softmax(1) * selected_outputs.log_softmax(1)).sum(1)
                    coeff = 1.0 / torch.exp(selected_entropy.detach() - current_threshold)
                    loss_pred = (selected_entropy * coeff).mean()
                else:
                    loss_pred = torch.tensor(0.0, device=self.device)

                total_loss = self.lambda_pre * loss_pred + cluster_loss

                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            with torch.no_grad():
                logits_eval, _ = self.wrapper(batch)
                logits_eval = logits_eval.detach()
            _, preds_eval = torch.max(logits_eval, 1)

            all_preds.append(preds_eval)
            all_labels.append(labels)

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'pred_loss': f'{loss_pred.item():.4f}',
                'cluster_loss': f'{cluster_loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}',
                'threshold': f'{current_threshold:.4f}'
            })

        all_preds_tensor = torch.cat(all_preds)
        all_labels_tensor = torch.cat(all_labels)

        epoch_evaluator = BinaryClassificationMetric(device=self.device)
        epoch_evaluator.update(all_preds_tensor, all_labels_tensor)
        epoch_metrics = epoch_evaluator.compute()

        logger.info(f"{Fore.CYAN}Per-Batch Aggregated Metrics for Epoch {epoch}:{Style.RESET_ALL}")
        logger.info(f"Acc: {epoch_metrics['acc']:.5f}, Macro F1: {epoch_metrics['macro_f1']:.5f}, "
                    f"Prec: {epoch_metrics['macro_prec']:.5f}, Rec: {epoch_metrics['macro_rec']:.5f}")
            
        wandb.log({
            'Acc': epoch_metrics['acc'],
            'F1': epoch_metrics['macro_f1'],
            'Prec': epoch_metrics['macro_prec'],
            'Rec': epoch_metrics['macro_rec'],
        })
        

@hydra.main(version_base=None, config_path="config", config_name="EN2MM")
def main(cfg: DictConfig):
    if not hasattr(cfg, 'task'):
        OmegaConf.set_struct(cfg, False)
        OmegaConf.update(cfg, "task", "binary")
        OmegaConf.set_struct(cfg, True)
        
    run = wandb.init(
        project='EN2MM', 
        dir=log_path,
        name=f"src/wandb/EN2MM/{datetime.now().strftime('%m%d-%H%M%S')}",
        config={
        'dataset': cfg.dataset,
        'model': cfg.model,
        'lr': cfg.opt.lr,
        'batch_size': cfg.batch_size,
        'all_config': OmegaConf.to_yaml(cfg),
        'task': cfg.task
        }
        )
    tags = []
    if OmegaConf.select(cfg, 'para.ablation') is not None:
        tags.append("ablation")
        wandb.config.update({'ablation': cfg.para.ablation})
    if OmegaConf.select(cfg, 'para.router_layer') is not None:
        wandb.config.update({'router_layer': cfg.para.router_layer})
    if OmegaConf.select(cfg, 'data.num_pos') is not None:  
        wandb.config.update({
            'num_pos': cfg.data.num_pos,
            'num_neg': cfg.data.num_neg
            })
    if cfg.task == 'ternary':
        tags.append("ternary")
    if hasattr(cfg, 'tag'):
        tags.append(cfg.tag)
    if OmegaConf.select(cfg, 'general') is not None:
        tags.append("general")
    run.tags = tags
    
    logger.remove()
    logger.add(log_path / 'log.log', retention="10 days", level="DEBUG")
    logger.add(sys.stdout, level="INFO")
    logger.info(OmegaConf.to_yaml(cfg))
    pd.set_option('future.no_silent_downcasting', True)
    colorama.init()
    set_seed(cfg.seed)
    
    trainer = Trainer(cfg)
    trainer.run()
    
    wandb.finish()

if  __name__ == '__main__':
    main()
