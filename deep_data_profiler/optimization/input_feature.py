import torch
import numpy as np
from enum import Enum, auto


class InputFeature:
    """
    InputFeature is optimized in the feature visualization. It uses 
        1. A type of feature visualization (currently, images parameterized either 
            in the pixel or Fourier basis).
        2. A torch Tensor, `fv_tensor`, that instantiates the feature, as well as initial parameters for 
            the optimization.
        3. A transformation. The transformation is used on the `fv_tensor`, and 
            mainly enforces a transformational robustness prior.
    Attributes
    ----------
    fv_type : Enum
        Type of feature visualization
    fv_tensor : torch.Tensor
        The feature visualization object that is optimized
    transform : Optional[Callable[[torch.Tensor], torch.Tensor]]
        The transformation made on the fv_tensor during optimization, e.g.,
        a scale transform.
    """

    def __init__(self, fv_type, dims=(1, 3, 224, 224), device=None):
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if fv_type == FeatureVizType.FFT_IMAGE:
            self.fv_type = FeatureVizType.FFT_IMAGE
            tensor, scale = initialize_fft_image(dims, device=self.device)
            self.transform = compose(
                linear_decorrelate, fft_transform(scale, dims)
            )
            self.fv_tensor = tensor

        elif fv_type == FeatureVizType.RGB_IMAGE:
            self.fv_type = FeatureVizType.RGB_IMAGE
            self.fv_tensor = (
                (torch.randn(*dims) * 0.01).to(self.device).requires_grad_(True)
            )
            self.transform = linear_decorrelate

        else:
            raise (
                NotImplementedError(
                    f"Have not implemented feature visualization type {fv_type}"
                )
            )

    def __call__(self, **kwargs):
        return self.transform(self.fv_tensor, **kwargs)

    def __repr__(self):
        return f"FV of type {self.fv_type}, with transform {self.transform}. \n {self.fv_tensor}"


class FeatureVizType(Enum):
    RGB_IMAGE = auto()
    FFT_IMAGE = auto()


def inverse_fft(image):
    scaled_spectrum_t = scale * spectrum_real_imag_t


def compose(g, f):
    def h(x):
        return g(f(x))

    return h


# From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def fft_transform(scale, dims=(1, 3, 224, 224)):
    def fft_inner(tensor):
        batch, channels, h, w = dims
        tensor = scale * tensor
        if type(tensor) is not torch.complex64:
            tensor = torch.view_as_complex(tensor)
        tensor = torch.fft.irfftn(tensor, s=(h, w), norm="ortho")
        tensor = tensor[:batch, :channels, :h, :w]
        tensor = tensor / 4.0  # Lucid constant that seems to reduce saturation
        return tensor

    return fft_inner


def initialize_fft_image(dims=(1, 3, 224, 224), device=None):
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch, channels, h, w = dims
    freqs = rfft2d_freqs(h, w)
    init_val_size = (
        (batch, channels) + freqs.shape + (2,)
    )  # 2 for imaginary and real components

    tensor = (
        (torch.randn(*init_val_size) * 0.01).to(device).requires_grad_(True)
    )

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h))
    scale = torch.tensor(scale).float()[None, None, ..., None].to(device)

    return tensor, scale


COLOR_CORRELATION_SVD_SQRT = np.asarray(
    [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
).astype("float32")
MAX_NORM_SVD_SQRT = np.max(np.linalg.norm(COLOR_CORRELATION_SVD_SQRT, axis=0))
COLOR_CORRELATION_NORMALIZED = COLOR_CORRELATION_SVD_SQRT / MAX_NORM_SVD_SQRT


def linear_decorrelate(tensor, device=None):
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_permute = tensor.permute(0, 2, 3, 1)
    t_permute = torch.matmul(
        t_permute, torch.tensor(COLOR_CORRELATION_NORMALIZED.T).to(device)
    )
    tensor = t_permute.permute(0, 3, 1, 2)
    return torch.sigmoid(tensor)
