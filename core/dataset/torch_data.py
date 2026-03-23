# dataset/torch_data.py
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class StandardScaler:
    def __init__(self, eps: float = 1e-8):
        self.mean_ = None
        self.std_ = None
        self.eps = eps

    def fit(self, X: np.ndarray):
        # X: (N, L, d)
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0) + self.eps
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler not fitted.")
        return (X - self.mean_) / self.std_


def load_npz(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


class TunnelWindowDataset(Dataset):
    """
    returns:
      x: (L,d) float32
      y_event_cls: scalar int64 class id
    """
    def __init__(self, npz_dict: dict, scaler: StandardScaler | None = None):
        X = npz_dict["X"].astype(np.float32)
        if "Y_event_cls" in npz_dict:
            Ye_cls = npz_dict["Y_event_cls"].astype(np.int64)
        elif "Y_event" in npz_dict:
            # Backward-compatible fallback from binary event labels.
            Ye_cls = npz_dict["Y_event"].astype(np.int64)
        else:
            raise KeyError("NPZ must contain 'Y_event_cls' (or legacy 'Y_event').")

        if scaler is not None:
            X = scaler.transform(X).astype(np.float32)

        self.X = X
        self.Ye_cls = Ye_cls

        self.feature_names = npz_dict.get("feature_names", npz_dict.get("feature_tags", None))
        self.event_class_names = npz_dict.get("event_class_names", None)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        y_event_cls = torch.tensor(self.Ye_cls[idx], dtype=torch.long)
        return x, y_event_cls


def build_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    train_npz = load_npz(train_path)
    val_npz = load_npz(val_path)
    test_npz = load_npz(test_path)

    scaler = StandardScaler().fit(train_npz["X"].astype(np.float32))

    train_ds = TunnelWindowDataset(train_npz, scaler=scaler)
    val_ds = TunnelWindowDataset(val_npz, scaler=scaler)
    test_ds = TunnelWindowDataset(test_npz, scaler=scaler)

    print(
        f"[shape] train X={train_ds.X.shape}, Y_event_cls={train_ds.Ye_cls.shape}; "
        f"val X={val_ds.X.shape}, Y_event_cls={val_ds.Ye_cls.shape}; "
        f"test X={test_ds.X.shape}, Y_event_cls={test_ds.Ye_cls.shape}"
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    meta = {
        "feature_names": train_ds.feature_names,
        "event_class_names": train_ds.event_class_names,
    }
    return train_loader, val_loader, test_loader, scaler, meta
