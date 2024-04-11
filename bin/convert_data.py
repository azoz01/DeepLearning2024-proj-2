import gc
import json
import numpy as np
import pickle as pkl
import pytorch_lightning as pl
import torch
import torchaudio

from loguru import logger
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from torchaudio.transforms import MelSpectrogram
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

SAMPLES = ["train", "val", "test"]
INPUT_DATA_PATH = Path("data/raw/train")
CONVERTED_DATA_PATH = Path("data/converted")
UNLABELED_DATA_PATH = Path("data/raw/test/audio")

VOICE_COMMANDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]

LABELED_DATA_COUNT = 64720
VALIDATION_DATA_COUNT = 6798
TESTING_DATA_COUNT = 6835
TRAINING_DATA_COUNT = LABELED_DATA_COUNT - (
    VALIDATION_DATA_COUNT + TESTING_DATA_COUNT
)

AVERAGE_SOUNDS_PER_CLASS = {
    "train": TRAINING_DATA_COUNT // 21,
    "val": VALIDATION_DATA_COUNT // 21,
    "test": TESTING_DATA_COUNT // 21,
}

GENERATION_LENGTHS_FRACTIONS = {
    "train": (0, TRAINING_DATA_COUNT / LABELED_DATA_COUNT),
    "val": (
        TRAINING_DATA_COUNT / LABELED_DATA_COUNT,
        (TRAINING_DATA_COUNT + VALIDATION_DATA_COUNT) / LABELED_DATA_COUNT,
    ),
    "test": (
        (TRAINING_DATA_COUNT + VALIDATION_DATA_COUNT) / LABELED_DATA_COUNT,
        1,
    ),
}


def main():
    pl.seed_everything(123)
    logger.info("Loading samples labels")
    with open(INPUT_DATA_PATH / "validation_list.txt", "r") as f:
        val_list = list(f)
    val_list = [el[:-1] for el in val_list]

    with open(INPUT_DATA_PATH / "testing_list.txt", "r") as f:
        test_list = list(f)
    test_list = [el[:-1] for el in test_list]

    logger.info("Initializing transformation")
    spectrogram_transform = MelSpectrogram(
        sample_rate=16_000, n_fft=321, n_mels=80
    )

    logger.info("Converting labeled data")
    outputs = {
        sample: {
            "data": [],
            "labels": [],
        }
        for sample in SAMPLES
    }

    labeled_data_paths = list(sorted(Path("data/raw/train/audio").iterdir()))
    for label_path in (pbar := tqdm(labeled_data_paths)):
        pbar.set_description(label_path.stem)
        label = label_path.name
        if label not in VOICE_COMMANDS and label != "_background_noise_":
            label = "unknown"

        if label == "_background_noise_":
            for sound_path in sorted(label_path.glob("*.wav")):
                waveform, _ = torchaudio.load(sound_path, normalize=True)
                for sample in ["train", "val", "test"]:
                    n_sounds = AVERAGE_SOUNDS_PER_CLASS[sample] // 6
                    n_start, n_end = GENERATION_LENGTHS_FRACTIONS[sample]
                    n_start = int(n_start * waveform.shape[1])
                    n_end = int(n_end * waveform.shape[1])
                    for _ in range(n_sounds):
                        sound_start = np.random.choice(
                            np.arange(n_start, n_end - 16_000)
                        )
                        sound_end = sound_start + 16_000
                        sound = waveform[0, sound_start:sound_end].unsqueeze(0)
                        spec = spectrogram_transform(sound)
                        spec = spec.permute(0, 2, 1)
                        spec = spec.squeeze(0)
                        outputs[sample]["data"].append(spec)
                        outputs[sample]["labels"].append("silence")
        else:
            for sound_path in sorted(label_path.glob("*.wav")):
                waveform, _ = torchaudio.load(sound_path, normalize=True)

                spec = spectrogram_transform(waveform)
                spec = spec.permute(0, 2, 1)
                spec = spec.squeeze(0)

                relative_path = "/".join(str(sound_path).split("/")[-2:])
                if relative_path in val_list:
                    sample = "val"
                elif relative_path in test_list:
                    sample = "test"
                else:
                    sample = "train"

                entry_to_append = outputs[sample]

                entry_to_append["data"].append(spec)
                entry_to_append["labels"].append(label)

    logger.info("Saving labeled data")
    if not CONVERTED_DATA_PATH.exists():
        CONVERTED_DATA_PATH.mkdir(exist_ok=True, parents=True)

    encoder = LabelEncoder()
    encoder.fit(outputs["train"]["labels"])
    with open(CONVERTED_DATA_PATH / "encoder.pkl", "wb") as f:
        pkl.dump(encoder, f)
    for sample in tqdm(SAMPLES):
        sample_data = outputs[sample]
        padded_data = pad_sequence(sample_data["data"], batch_first=True)
        print(padded_data.shape)
        torch.save(padded_data, CONVERTED_DATA_PATH / f"{sample}_data.pt")
        if sample != "noise":
            with open(CONVERTED_DATA_PATH / f"{sample}_labels.json", "w") as f:
                json.dump(encoder.transform(sample_data["labels"]).tolist(), f)

    logger.info("Clearning memory")
    del outputs
    gc.collect()

    logger.info("Converting unlabeled data")
    unlabeled_sounds_path = list(sorted(UNLABELED_DATA_PATH.iterdir()))
    unlabeled_output = []
    for sound_path in tqdm(unlabeled_sounds_path):
        waveform, _ = torchaudio.load(sound_path, normalize=True)

        spec = spectrogram_transform(waveform)
        spec = spec.permute(0, 2, 1)
        spec = spec.squeeze(0)
        unlabeled_output.append(spec)

    logger.info("Saving unlabeled data")
    unlabeled_output = pad_sequence(unlabeled_output)
    torch.save(unlabeled_output, CONVERTED_DATA_PATH / "unlabeled_data.pt")


if __name__ == "__main__":
    main()
