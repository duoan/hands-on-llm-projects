import time
from typing import Dict
import random
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, ViTModel, get_cosine_schedule_with_warmup

# Constants
BATCH_SIZE = 256
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 300
TEMPERATURE = 0.1
IMAGE_SIZE = 224  # ViT requires 224x224 input
EMBEDDING_DIM = 256
WARMUP_STEPS = 1000

# CIFAR10 label names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# use multiple text template to enhance the text diversity
TEXT_TEMPLATES = [
    "a photo of a {}.",
    "an image of a {}.",
    "this is a picture of a {}.",
    "a {} in the image.",
    "{} in a photograph."
]


class CIFAR10CLIPDataset(Dataset):
    def __init__(self, train: bool = True):
        # the preprocess of image
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(2, 9),  # 使用RandAugment
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # load CIFAR10 dataset
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )

        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        image, label = self.dataset[idx]
        template = random.choice(TEXT_TEMPLATES)
        text = template.format(CIFAR10_CLASSES[label])

        # tokenize text
        text_inputs = self.tokenizer(
            text,
            padding='max_length',
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )

        return {
            'pixel_values': image,
            'input_ids': text_inputs['input_ids'].squeeze(),
            'attention_mask': text_inputs['attention_mask'].squeeze()
        }


class CLIPModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Initialize encoders
        self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')

        # project layers
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_encoder.config.hidden_size, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, EMBEDDING_DIM)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, EMBEDDING_DIM)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(pixel_values).last_hidden_state[:, 0, :]
        image_features = self.image_projection(image_features)
        return F.normalize(image_features, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_features = self.text_encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_features)
        return F.normalize(text_features, dim=-1)

    def forward(self, batch):
        # extract the image and text features
        image_features = self.encode_image(batch['pixel_values'])
        text_features = self.encode_text(batch['input_ids'], batch['attention_mask'])

        # calculate the similarity of image and text
        logits = torch.matmul(image_features, text_features.T) / TEMPERATURE

        # create labels
        labels = torch.arange(len(logits), device=logits.device)

        # calculate the double loss
        loss_i2t = F.cross_entropy(logits, labels, label_smoothing=0.1)
        loss_t2i = F.cross_entropy(logits.T, labels, label_smoothing=0.1)

        return (loss_i2t + loss_t2i) / 2

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss)

        # Calculate the accuracy
        with torch.no_grad():
            image_features = self.encode_image(batch['pixel_values'])
            text_features = self.encode_text(batch['input_ids'], batch['attention_mask'])
            logits = torch.matmul(image_features, text_features.T)
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == torch.arange(len(preds), device=preds.device)).float().mean()
            self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('val_loss', loss)

        # Calculate the accuracy
        image_features = self.encode_image(batch['pixel_values'])
        text_features = self.encode_text(batch['input_ids'], batch['attention_mask'])
        logits = torch.matmul(image_features, text_features.T)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == torch.arange(len(preds), device=preds.device)).float().mean()
        self.log('val_acc', acc)

        return loss

    def configure_optimizers(self):
        # learn with different learning rate for each layers
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in self.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # warmup with cosine decay
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def main():
    torch.set_float32_matmul_precision("high")
    # Create dataset
    train_dataset = CIFAR10CLIPDataset(train=True)
    val_dataset = CIFAR10CLIPDataset(train=False)

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True
    )

    tb_logger = TensorBoardLogger(
        save_dir='logs',
        name='clip_training',
        version=time.strftime("%Y%m%d-%H%M%S")
    )

    csv_logger = CSVLogger(
        save_dir='logs',
        name='clip_metrics'
    )

    # Init model and trainer
    model = CLIPModel()
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='gpu',
        devices=1,
        precision='bf16-mixed',
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        logger=[csv_logger, tb_logger],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                dirpath='checkpoints',
                filename='clip-cifar10-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                mode='min'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                mode='min',
                min_delta=0.001,
                verbose=True
            ),
            pl.callbacks.DeviceStatsMonitor(),
            pl.callbacks.Timer(interval="epoch", verbose=True),
            pl.callbacks.TQDMProgressBar(refresh_rate=10),
            pl.callbacks.RichModelSummary(max_depth=2),
        ]
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Print the best model
    print(f"Best model score: {trainer.checkpoint_callback.best_model_score}")
    print(f"Best model path: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
