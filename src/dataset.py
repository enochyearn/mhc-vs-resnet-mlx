import gzip
import shutil
import struct
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

import mlx.core as mx
import numpy as np


class FashionMNISTLoader:
    def __init__(self, data_dir=None, download=True):
        repo_root = Path(__file__).resolve().parents[1]
        self.data_dir = Path(data_dir) if data_dir else repo_root / "data"
        self.download_enabled = download
        self.base_urls = [
            "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        ]
        self.files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz",
        }
        self._data = None

    def _download_file(self, url, target, retries=2, timeout=10):
        last_error = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp, open(
                    target, "wb"
                ) as out:
                    shutil.copyfileobj(resp, out)
                return
            except (URLError, HTTPError, OSError) as exc:
                last_error = exc
                time.sleep(1 + attempt)
        raise last_error

    def download(self):
        if not self.download_enabled:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for name in self.files.values():
            target = self.data_dir / name
            if target.exists():
                continue
            last_error = None
            for base_url in self.base_urls:
                url = base_url + name
                try:
                    self._download_file(url, target)
                    last_error = None
                    break
                except (URLError, HTTPError, OSError) as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error

    def _ensure_files(self):
        missing = [name for name in self.files.values() if not (self.data_dir / name).exists()]
        if not missing:
            return
        if not self.download_enabled:
            raise FileNotFoundError(
                f"Missing Fashion-MNIST files in {self.data_dir}. "
                "Enable download or place the files manually."
            )
        self.download()

    def _load_images(self, path):
        with gzip.open(path, "rb") as f:
            magic = struct.unpack(">I", f.read(4))[0]
            if magic != 2051:
                raise ValueError(f"Invalid image file magic: {magic}")
            count, rows, cols = struct.unpack(">III", f.read(12))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(count, rows, cols).astype(np.float32) / 255.0
        return data

    def _load_labels(self, path):
        with gzip.open(path, "rb") as f:
            magic = struct.unpack(">I", f.read(4))[0]
            if magic != 2049:
                raise ValueError(f"Invalid label file magic: {magic}")
            count = struct.unpack(">I", f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(count).astype(np.int64)

    def load(self):
        if self._data is not None:
            return self._data

        self.data_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.data_dir / "fashion-mnist.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            self._data = (
                mx.array(cached["x_train"]),
                mx.array(cached["y_train"]),
                mx.array(cached["x_test"]),
                mx.array(cached["y_test"]),
            )
            return self._data

        self._ensure_files()

        x_train = self._load_images(self.data_dir / self.files["train_images"])
        y_train = self._load_labels(self.data_dir / self.files["train_labels"])
        x_test = self._load_images(self.data_dir / self.files["test_images"])
        y_test = self._load_labels(self.data_dir / self.files["test_labels"])

        np.savez(
            cache_path,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

        self._data = (
            mx.array(x_train),
            mx.array(y_train),
            mx.array(x_test),
            mx.array(y_test),
        )
        return self._data

    def get_batches(self, batch_size, split="train", shuffle=True, seed=None, repeat=False):
        x_train, y_train, x_test, y_test = self.load()
        if split == "train":
            x_data, y_data = x_train, y_train
        elif split == "test":
            x_data, y_data = x_test, y_test
        else:
            raise ValueError("split must be 'train' or 'test'")

        rng = np.random.default_rng(seed)
        while True:
            indices = np.arange(x_data.shape[0])
            if shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_idx = mx.array(batch_idx)
                yield x_data[batch_idx], y_data[batch_idx]
            if not repeat:
                break
