import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from engine.attention import AttentionAudioClassifier
from engine.data import get_main_classes_loader

torch.set_float32_matmul_precision("high")
pl.seed_everything(123)

train_loader = get_main_classes_loader("train", oversample_silence=True)
val_loader = get_main_classes_loader("val")
test_loader = get_main_classes_loader("test")

EMBEDDING_SIZE = 80
for i in range(5):
    print(i)
    pl.seed_everything(i)
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
            every_n_epochs=1,
        ),
    ]

    model = AttentionAudioClassifier(3, 100, EMBEDDING_SIZE, 512, 4, 4).cuda()
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=callbacks,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        default_root_dir="results/attention_main_classes",
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)
