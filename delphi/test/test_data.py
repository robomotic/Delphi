import numpy as np
from dacite import from_dict

from delphi.data.dataset import Dataset, UKBDataConfig, get_p2i, sort_by_time, squeeze
from delphi.model.transformer import Delphi, DelphiConfig
from delphi.utils import get_batch

test_cfg = {
    "train_data": {
        "data_dir": "data/ukb_simulated_data",
        "memmap_fname": "val.bin",
        "tokenizer_fname": "tokenizer.yaml",
        "transforms": [
            {
                "name": "no-event",
                "args": {
                    "interval_in_years": 5,
                    "mode": "regular",
                    "max_age_in_years": 100,
                },
            },
            {"name": "augment-lifestyle", "args": {}},
        ],
    }
}

test_cfg = from_dict(
    data_class=UKBDataConfig,
    data=test_cfg["train_data"],
)

val = np.fromfile("data/ukb_simulated_data/val.bin", dtype=np.uint32).reshape(-1, 3)
val_p2i = get_p2i(val)
batch_idx = np.arange(128)

a_X_t0, a_T_t0, _, _ = get_batch(
    batch_idx,
    val,
    val_p2i,
    select="left",
    block_size=1000,
    device="cpu",
    padding="regular",
    no_event_token_rate=5,
    cut_batch=True,
    lifestyle_augmentations=False,
)


dataset = Dataset(cfg=test_cfg)
_, b_X, b_T, _, _, _ = dataset.get_batch(batch_idx)
b_X, b_T = sort_by_time(b_X, b_T)
b_X, b_T = squeeze(b_X, b_T)
b_X_t0, b_X_t1, b_T_t0, b_T_t1 = b_X[:, :-1], b_X[:, 1:], b_T[:, :-1], b_T[:, 1:]

print("debug")
