import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


def _norm_path(p: str) -> str:
    """Normalize Windows-style backslash paths from CSV to POSIX."""
    return p.replace("\\", "/").replace("./", "", 1)


class LoadData(Dataset):
    """
    Audiovisual dataset (fully in-memory).

    Accepts either a CSV file path or a pre-loaded DataFrame (for in-memory
    train/val splits without writing temp files).

    When config.use_multi_audio=True, concatenates ECAPA-TDNN and WavLM-SV
    features to form a richer audio representation.
    """

    def __init__(
        self,
        config,
        audio_encoder: str = "ecappa_feats_path",
        audio_encoder2: str = None,
        csv_path: str = None,
        df: pd.DataFrame = None,
    ):
        assert (csv_path is not None) ^ (df is not None), (
            "Provide exactly one of csv_path or df."
        )

        if df is None:
            df = pd.read_csv(csv_path)

        self.num_samples = len(df)

        audio_paths = [
            str(Path(config.home_dir) / _norm_path(p))
            for p in df[audio_encoder]
        ]
        audio2_paths = (
            [str(Path(config.home_dir) / _norm_path(p)) for p in df[audio_encoder2]]
            if audio_encoder2 else None
        )
        face_paths = [
            str(Path(config.home_dir) / _norm_path(p))
            for p in df["facenet_feats_path"]
        ]
        labels = df["label"].astype(int).to_numpy()

        audio_feats, face_feats = [], []
        for i in range(self.num_samples):
            a = np.load(audio_paths[i]).astype("float32")
            if audio2_paths is not None:
                a2 = np.load(audio2_paths[i]).astype("float32")
                a = np.concatenate([a, a2], axis=0)
            audio_feats.append(a)
            face_feats.append(np.load(face_paths[i]).astype("float32"))

        self.audio_feats = np.stack(audio_feats)   # (N, Da) or (N, Da1+Da2)
        self.face_feats  = np.stack(face_feats)    # (N, Df)
        self.labels      = labels                  # (N,)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.audio_feats[idx],
            self.face_feats[idx],
            self.labels[idx],
        )
