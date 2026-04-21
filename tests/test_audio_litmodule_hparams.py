import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

speechbrain = types.ModuleType("speechbrain")
processing = types.ModuleType("speechbrain.processing")
speech_augmentation = types.ModuleType("speechbrain.processing.speech_augmentation")


class DummySpeedPerturb:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, tensor):
        return tensor


speech_augmentation.SpeedPerturb = DummySpeedPerturb
processing.speech_augmentation = speech_augmentation
speechbrain.processing = processing

sys.modules.setdefault("speechbrain", speechbrain)
sys.modules.setdefault("speechbrain.processing", processing)
sys.modules.setdefault("speechbrain.processing.speech_augmentation", speech_augmentation)

from look2hear.system.audio_litmodule import AudioLightningModule


def test_config_to_hparams_keeps_string_lists_and_converts_numeric_lists():
    config = {
        "datamodule": {"data_config": {"sample_rate": 16000}},
        "training": {"SpeedAug": False},
        "audionet": {
            "audionet_config": {
                "binary_config": {
                    "protect_patterns": ["bandsplit.proj", "q_proj"],
                    "kernel_sizes": [3, 5, 7],
                }
            }
        },
    }

    hparams = AudioLightningModule.config_to_hparams(config)

    assert hparams["audionet_audionet_config_binary_config_protect_patterns"] == [
        "bandsplit.proj",
        "q_proj",
    ]
    assert torch.equal(
        hparams["audionet_audionet_config_binary_config_kernel_sizes"],
        torch.tensor([3, 5, 7]),
    )
