import pytorch_lightning as pl
import torch

from loguru import logger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from engine.attention import AttentionAudioClassifier
from engine.data import get_data_loader

torch.set_float32_matmul_precision("high")
pl.seed_everything(123)

EMBEDDING_SIZE = 80

def main():
    logger.info("Loading data")
    train_loader = get_data_loader("train", undersample_majority=True)
    val_loader = get_data_loader("val")
    test_loader = get_data_loader("test")

    logger.info("Initializing callbacks")
    callbacks = [
        EarlyStopping(
            monitor="val_balanced_accuracy",
            mode="max",
            patience=10,
            min_delta=1e-5,
        ),
        ModelCheckpoint(
            save_top_k=1,
            monitor="val_accuracy",
            filename="model-{epoch}-{val_accuracy:.2f}",
            mode="max",
            every_n_epochs=1
        )
    ]
    
    logger.info("Initializing model & trainer")
    model = AttentionAudioClassifier(12, 100, EMBEDDING_SIZE, 512, 4, 4).cuda()
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=callbacks,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm"
    )
    
    logger.info("Training model")
    trainer.fit(model, train_loader, val_loader)
    

if __name__ == "__main__":
    main()