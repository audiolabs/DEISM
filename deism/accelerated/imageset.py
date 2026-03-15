from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ImageSet:
    """Normalized image/reflection container for both shoebox and ARG paths."""

    A: Optional[np.ndarray]
    R_sI_r_all: np.ndarray
    atten_all: np.ndarray
    R_s_rI_all: Optional[np.ndarray] = None
    R_r_sI_all: Optional[np.ndarray] = None
    early_indices: Optional[np.ndarray] = None
    late_indices: Optional[np.ndarray] = None

    @property
    def n_images(self) -> int:
        return int(self.R_sI_r_all.shape[0])

    def validate(self) -> None:
        if self.R_sI_r_all.ndim != 2 or self.R_sI_r_all.shape[1] != 3:
            raise ValueError("R_sI_r_all must be shape [n_images, 3].")
        if self.atten_all.ndim != 2:
            raise ValueError("atten_all must be shape [n_images, n_freqs].")
        if self.atten_all.shape[0] != self.n_images:
            raise ValueError("atten_all image axis does not match R_sI_r_all.")
        if self.A is not None and self.A.shape[0] != self.n_images:
            raise ValueError("A image axis does not match R_sI_r_all.")
        if self.R_s_rI_all is not None and self.R_s_rI_all.shape[0] != self.n_images:
            raise ValueError("R_s_rI_all image axis does not match R_sI_r_all.")
        if self.R_r_sI_all is not None and self.R_r_sI_all.shape[0] != self.n_images:
            raise ValueError("R_r_sI_all image axis does not match R_sI_r_all.")

    @staticmethod
    def from_shoebox_images(images: Dict[str, np.ndarray]) -> "ImageSet":
        data = ImageSet(
            A=np.asarray(images["A"]),
            R_sI_r_all=np.asarray(images["R_sI_r_all"]),
            atten_all=np.asarray(images["atten_all"]),
            R_s_rI_all=np.asarray(images.get("R_s_rI_all"))
            if "R_s_rI_all" in images
            else None,
            R_r_sI_all=np.asarray(images.get("R_r_sI_all"))
            if "R_r_sI_all" in images
            else None,
        )
        data.validate()
        return data

    @staticmethod
    def from_arg_images(images: Dict[str, np.ndarray]) -> "ImageSet":
        # ARG image data is often stored with shape [3, n_images] / [n_freqs, n_images].
        r_sI = np.asarray(images["R_sI_r_all"])
        if r_sI.ndim != 2:
            raise ValueError("R_sI_r_all for ARG must be 2-D.")
        if r_sI.shape[0] == 3:
            r_sI = r_sI.T
        atten = np.asarray(images["atten_all"])
        if atten.ndim != 2:
            raise ValueError("atten_all for ARG must be 2-D.")
        if atten.shape[0] != r_sI.shape[0]:
            atten = atten.T

        data = ImageSet(
            A=None,
            R_sI_r_all=r_sI,
            atten_all=atten,
            early_indices=np.asarray(images.get("early_indices"))
            if "early_indices" in images
            else None,
            late_indices=np.asarray(images.get("late_indices"))
            if "late_indices" in images
            else None,
        )
        data.validate()
        return data

    def to_shoebox_legacy(self) -> Dict[str, np.ndarray]:
        out: Dict[str, Any] = {
            "R_sI_r_all": self.R_sI_r_all,
            "atten_all": self.atten_all,
        }
        if self.A is not None:
            out["A"] = self.A
        if self.R_s_rI_all is not None:
            out["R_s_rI_all"] = self.R_s_rI_all
        if self.R_r_sI_all is not None:
            out["R_r_sI_all"] = self.R_r_sI_all
        if self.early_indices is not None:
            out["early_indices"] = self.early_indices
        if self.late_indices is not None:
            out["late_indices"] = self.late_indices
        return out

    def to_arg_legacy(self) -> Dict[str, np.ndarray]:
        out: Dict[str, Any] = {
            "R_sI_r_all": self.R_sI_r_all.T,
            "atten_all": self.atten_all.T,
        }
        if self.early_indices is not None:
            out["early_indices"] = self.early_indices
        if self.late_indices is not None:
            out["late_indices"] = self.late_indices
        return out
