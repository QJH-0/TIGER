import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from look2hear.datas import Libri2MixModuleRemix


def test_librimix_datamodule_respects_loader_memory_flags():
    datamodule = Libri2MixModuleRemix(
        train_dir="train",
        valid_dir="valid",
        test_dir="test",
        batch_size=2,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False,
    )
    datamodule.data_train = [1, 2]
    datamodule.data_val = [3]
    datamodule.data_test = [4]

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert train_loader.pin_memory is False
    assert val_loader.pin_memory is False
    assert test_loader.pin_memory is False
    assert train_loader.persistent_workers is False
    assert val_loader.persistent_workers is False
    assert test_loader.persistent_workers is False


def test_librimix_datamodule_disables_persistent_workers_when_num_workers_is_zero():
    datamodule = Libri2MixModuleRemix(
        train_dir="train",
        valid_dir="valid",
        test_dir="test",
        batch_size=2,
        num_workers=0,
        pin_memory=True,
        persistent_workers=True,
    )
    datamodule.data_train = [1, 2]

    train_loader = datamodule.train_dataloader()

    assert train_loader.persistent_workers is False
    assert train_loader.pin_memory is True
