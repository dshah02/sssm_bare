import torch

def get_hankel(seq_len: int, use_hankel_L: bool = False) -> torch.Tensor:
    indices = torch.arange(1, seq_len + 1)
    hankel = indices[:, None] + indices[None, :]

    if use_hankel_L:
        sgn = -(1.0 ** (hankel - 2.0)) + 1.0
        denom = (hankel + 3.0) * (hankel - 1.0) * (hankel + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (hankel**3 - hankel)

    return Z

def get_spectral_filters(
    seq_len: int, K: int, use_hankel_L: bool = False,
) -> torch.Tensor:
    assert torch.cuda.is_available(), "CUDA is required."
    device = torch.device("cuda")
    Z = get_hankel(seq_len, use_hankel_L).to(device)
    sigma, phi = torch.linalg.eigh(Z)
    sigma, phi = sigma[-K:], phi[:, -K:]
    phi *= sigma
    return phi

def preconvolve(phi: torch.Tensor, n: int, approx: bool = True) -> tuple[torch.Tensor, int]:
    seq_len, K = phi.shape
    phi = phi.view(1, seq_len, K, 1)
    signal = torch.fft.rfft(phi, n=n, dim=1)
    return signal

def convolve(u: torch.Tensor, v: torch.Tensor, n: int, approx: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, d_in = u.shape

    sgn = torch.full((1, seq_len, 1), 1, device=u.device)
    sgn[:, 1::2] *= -1
    if approx:
        _, d_out = v.shape
        v = v.view(1, seq_len, d_out, 1).to(torch.float32)
    else:
        _, K = v.shape
        sgn = sgn.unsqueeze(-1)
        v = v.view(1, seq_len, K, 1, 1).to(torch.float32)
        u = u.view(bsz, seq_len, 1, d_in).expand(bsz, seq_len, K, d_in)

    v = torch.fft.rfft(v, n=n, dim=1)
    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32)
    U = torch.fft.rfft(U, n=n, dim=1)
    U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)
    U_minus = U_minus * sgn

    return U_plus, U_minus

def nearest_power_of_2(x: int) -> int:
    """
    Returns the smallest power of 2 that is greater than or equal to x.
    If x is already a power of 2, it returns x itself.
    Otherwise, it returns the next higher power of 2.

    Args:
        x (int): The input integer.

    Returns:
        int: The smallest power of 2 that is greater than or equal to x.
    """
    s = bin(x)
    s = s.lstrip("-0b")
    length = len(s)
    return 1 << (length - 1) if x == 1 << (length - 1) else 1 << length

def conv(u: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implements the FFT convolution of the input sequences into the Hankel
    spectral basis, as described in Section 3 of the paper.

    This function computes U⁺_{t,k} and U⁻_{t,k}, which are the positive and
    negative featurizations of the input sequence, respectively.

    Args:
        u (torch.Tensor): Input of shape [bsz, sl, d].
        phi (torch.Tensor): Top K eigenvectors of shape [sl, K].

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Feature tensors U⁺ and U⁻ of shape [bsz, sl, K, d].
    """
    bsz, sl, d = u.shape
    _, K = phi.shape

    # Round sequence length to the nearest power of 2 for efficient convolution
    n = nearest_power_of_2(sl * 2 - 1)

    # Add bsz and d dims to phi and u and expand to the return shape
    phi = phi.view(1, -1, K, 1).expand(bsz, -1, K, d)
    u = u.view(bsz, -1, 1, d).expand(bsz, -1, K, d)

    # Compute U⁺
    V = torch.fft.rfft(phi, n=n, dim=1)
    U = torch.fft.rfft(u, n=n, dim=1)
    U_plus = torch.fft.irfft(V * U, n=n, dim=1)[:, :sl]

    # Generate alternating signs tensor, (-1)^i of length sl, match dims of u
    alt = torch.ones(sl, device=u.device)
    alt[1::2] = -1  # Replace every other element with -1, starting from index 1
    alt = alt.view(1, sl, 1, 1).expand_as(u)

    # Compute U⁻
    u_alt = u * alt
    U_alt = torch.fft.rfft(u_alt, n=n, dim=1)
    U_minus = torch.fft.irfft(V * U_alt, n=n, dim=1)[:, :sl]

    return U_plus, U_minus

# Additional functions
def shift(u: torch.Tensor, k: int = 1) -> torch.Tensor:
    if k == 0:
        return u
    shifted = torch.roll(u, shifts=k, dims=1)
    shifted[:, :k] = 0
    return shifted

def compute_ar_u(M_u: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
    k_u = M_u.shape[0]
    u_shifted = torch.stack([shift(u_t, i) for i in range(k_u)], dim=1)
    ar_u = torch.einsum("bksi,koi->bso", u_shifted, M_u)
    return ar_u

def compute_ar_y(M_y: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    d_out, k_y, _ = M_y.shape
    bsz, sl, _ = y_t.shape

    eye = torch.eye((k_y - 1) * d_out, k_y * d_out, dtype=y_t.dtype, device=y_t.device)
    A = M_y.reshape(d_out, k_y * d_out)
    A = torch.cat([A, eye], dim=0)
    A = A.unsqueeze(0).expand(bsz, k_y * d_out, k_y * d_out)

    padding = torch.zeros(bsz, sl, (k_y - 1) * d_out, dtype=y_t.dtype, device=y_t.device)
    state = torch.cat([y_t, padding], dim=2)
    state = state.view(bsz, sl, k_y * d_out, 1)

    y = state[:, 0]
    ys = [y[:, :d_out, 0]]

    for i in range(1, sl):
        y_next = state[:, i]
        y = torch.bmm(A, y) + y_next
        ys.append(y[:, :d_out, 0])

    return torch.stack(ys, dim=1)