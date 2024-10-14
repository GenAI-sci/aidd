

from dataclasses import dataclass, asdict
from collections import OrderedDict
from typing import Optional, Any, Dict
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F

from torch.distributed import init_process_group, destroy_process_group

from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from transformers import GPT2Config,GPT2LMHeadModel


#model.save_pretrained('mol_gpt')

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@dataclass
class TrainerConfig:
    max_epochs: int = None
    batch_size: int = None
    data_loader_workers: int = None
    grad_norm_clip: float = None
    save_every: int = None
    use_amp: bool = False



class Trainer:

    def __init__(self, trainer_config: TrainerConfig, model, optimizer, train_dataset, test_dataset=None):
        self.config = trainer_config
        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])  
        # data stuff
        self.train_dataset = train_dataset
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = self._prepare_dataloader(test_dataset) if test_dataset else None
        # initialize train states
        self.epochs_run = 0
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer        
        self.save_every = self.config.save_every
        if self.config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.model = DDP(self.model, device_ids=[self.local_rank])
        
    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
            sampler=DistributedSampler(dataset)
        )



    def _run_batch(self, source, targets, loss_mask, train: bool = True) -> float:
        with torch.set_grad_enabled(train):
            logits = self.model(source).logits
            logits=logits.view(-1,logits.size(-1))
          
            loss = F.cross_entropy(logits, targets.view(-1),reduction='none' )
           
            losses = loss.float()
            loss_mask = loss_mask.view(-1).float()
            
            loss = torch.sum(losses.view(-1) * loss_mask) / (loss_mask > 0).sum()
            

        if train:
            self.optimizer.zero_grad(set_to_none=True)
           
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
            self.optimizer.step()
        
        return loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        dataloader.sampler.set_epoch(epoch)
        for iter, (source, targets,loss_mask) in enumerate(dataloader):
            step_type = "Train" if train else "Eval"
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            loss_mask = loss_mask.to(self.local_rank)
            batch_loss = self._run_batch(source, targets,loss_mask, train)
            if iter % 100 == 0 and self.local_rank==0 :
                print(f"[GPU{self.global_rank}] Epoch {epoch} | Iter {iter} | {step_type} Loss {batch_loss:.5f}")

    def train(self):
        from datetime import datetime
        
        for epoch in range(self.epochs_run, self.config.max_epochs):
            epoch += 1
            start = datetime.now()
            self._run_epoch(epoch, self.train_loader, train=True)
            # eval run
            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)
            
            if self.local_rank==0 :
                difference = datetime.now() - start
                
                print(f'Epoch {epoch} rutime {difference.total_seconds()}')
                
   

class SmilesDataset(Dataset):
    def __init__(self, root_dir,max_length):
        self.samples = []
        with open(root_dir,'r') as f:
            for line in f.readlines():
                # as end
                self.samples.append(line.strip()+'<|endoftext|>')
        
        self.tokenizer=AutoTokenizer.from_pretrained('./gpt2tokenizer')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length=max_length
  

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        encoding = self.tokenizer(self.samples[idx], padding='max_length', truncation=True, return_tensors="pt",max_length=self.max_length)
        tokens=encoding.input_ids[0]
        loss_mask=encoding.attention_mask[0]
        
        # 0 as bos
        x = torch.cat([torch.tensor([0]),tokens])[:-1]
        y = tokens
       
        return x,y,loss_mask

                
if __name__=='__main__':
    ddp_setup()

    
    trainer_cfg = TrainerConfig()
   
    trainer_cfg.max_epochs=5
    
    trainer_cfg.batch_size = 32
    trainer_cfg.data_loader_workers=2
    trainer_cfg.grad_norm_clip=0.3
    trainer_cfg.save_every=1
    
    
    
    from transformers import GPT2Config,GPT2LMHeadModel
    config=GPT2Config()
    model=GPT2LMHeadModel(config)

       

    train_dataset=SmilesDataset('muv/train_smiles.txt',max_length=128)
    test_dataset=SmilesDataset('muv/train_smiles.txt',max_length=128)


    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, betas=(0.9, 0.95))
    trainer = Trainer(trainer_cfg, model, optimizer, train_dataset, test_dataset)
    trainer.train()
    
    if trainer.local_rank == 0:
        model.save_pretrained('mol_gpt')
                
    destroy_process_group()
