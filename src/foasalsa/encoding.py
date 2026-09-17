import torch

from .stft import StftConfig, stft, istft


def _gradient_matrix(mics: torch.Tensor, # [C, 3]
                    freqs: torch.Tensor, # [F]
                    speed_of_sound: float = 343.0,
                    rtol: float = 1e-5,
                    compensate: bool = True,
                    regularize: bool = True,
                    max_gain_db: float = 20.0,
) -> torch.Tensor: # [F, 4, C]
    """
    Build the frequency-dependent MIC-to-FOA encoding matrix
    using the first-order pressure-gradient approximation.

    Args:
        mics: [C, 3]
            C - number of microphones
            3 - microphone coordinates (x, y, z), in meters

        freqs: [F]
            F - number of frequency bins
            Contains the frequency in Hz corresponding to each STFT bin.

        speed_of_sound:
            Speed of sound in meters per second, used to compute
            the wavenumber k = 2*pi*f / c.

        rtol:
            Relative tolerance used by the pseudoinverse.
            Small singular values are treated as zero, preventing
            amplification of spatial directions that the microphone
            geometry cannot reliably observe.

        compensate:
            If True, remove the j*k factor introduced by the spatial
            pressure gradient so that XYZ approximate s * direction.
            If False, return the uncompensated gradient components.

        regularize:
            If True, regularize the 1/(j*k) compensation at low
            frequencies to avoid excessive noise amplification.

        max_gain_db:
            Maximum allowed low-frequency gradient amplification,
            in dB. Used to determine the regularization strength.

    Returns:
        E: [F, 4, C], complex
            F - number of frequency bins
            4 - FOA channels [W, X, Y, Z]
            C - number of microphone channels

            For each frequency, E[f] is a [4, C] matrix that mixes
            the C microphone signals into the four FOA channels.

            W is estimated as the average microphone pressure.
            XYZ are estimated from spatial pressure differences
            using the pseudoinverse of the centered microphone geometry.
    """
    C = mics.shape[0]
    F = freqs.shape[0]
    freqs = freqs.to(device=mics.device, dtype=mics.dtype)

    # E multiplies the complex STFT spectrum, so every branch below
    # has to end up complex; decide the precision once, up front
    complex_dtype = (
        torch.complex128 if mics.dtype == torch.float64 else torch.complex64
    )

    # center microphone coordinates around the array origin
    # makes the common pressure component vanish when multiplied by D_pinv
    D = mics - mics.mean(dim=0, keepdim=True) # [C, 3]

    # map micropfone pressure gradients [C] to spatial components [X, Y, Z]
    D_pinv = torch.linalg.pinv(D, rtol=rtol) # [3, C]

    # W is the avarage of pressure across mirophones
    w = torch.full(
        (F, 1, C),
        1.0 / C,
        device=mics.device,
        dtype=mics.dtype,
    ) # [F, 1, C]

    # spatial angular frequency (wavenumber)
    k = 2 * torch.pi * freqs / speed_of_sound # [F]

    if compensate:
        if regularize:
            # maximum microphone distance from the center
            radius = torch.linalg.vector_norm(D, dim=-1).max()

            if radius <= 0:
                raise ValueError("Microphone positions must not all be identical.")

            # maximum allowed low-frequency gain from dB to linear scale
            gain_max = 10 ** (max_gain_db / 20)

            # regularization strength determined by array size
            # and maximum allowed low-frequency gain
            k_reg = 1.0 / (2.0 * radius * gain_max)

            # regularized approximation of 1 / (j*k)
            # approaches 1/(j*k) at higher frequencies
            # goes to zero instead of infinity as f -> 0
            compensation = (-1j * k / (k.square() + k_reg**2))
        else:
            # exact 1/(j*k) compensation
            compensation = torch.zeros(
                F,
                device=k.device,
                dtype=complex_dtype,
            )

            # k = 0 at DC, where the gradient carries no direction at all
            mask = k != 0
            compensation[mask] = 1.0 / (1j * k[mask])

        # [F, 1, 1] * [3, C] -> [F, 3, C]
        xyz = compensation[:, None, None] * D_pinv
    else:
        # same spatial-gradient matrix for every frequency without compensation
        # [3, C] -> [F, 3, C]
        xyz = D_pinv.unsqueeze(0).expand(F, -1, -1)

    # E is applied to the complex STFT spectrum, so keep W and XYZ complex
    w = w.to(complex_dtype)
    xyz = xyz.to(complex_dtype)

    # [F, 1, C] + [F, 3, C] -> [F, 4, C]
    return torch.cat([w, xyz], dim=1)


def _fibonacci_sphere(
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Generate approximately uniform unit directions on a sphere.

    Args:
        n:
            Number of directions V.

        device:
            Device for the output tensor.

        dtype:
            Floating-point dtype for the output tensor.

    Returns:
        directions: [V, 3]
            V - number of directions
            3 - Cartesian components [x, y, z]

            Each row is a unit vector representing one direction
            on the sphere.
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    # point indices [V]
    i = torch.arange(n, device=device, dtype=dtype)

    # spread points uniformly from the north to the south pole
    z = 1.0 - 2.0 * (i + 0.5) / n # half-step offsets avoids placing points exactly at the poles

    # radius of the horizontal circle at each z coordinate
    radius = torch.sqrt(torch.clamp(1.0 - z.square(), min=0.0))

    # distributes consecutive points around the sphere
    # without regular latitude/longitude clusters
    golden_angle = torch.pi * (3.0 - torch.sqrt(
        torch.tensor(5.0, device=device, dtype=dtype)
    ))

    theta = i * golden_angle

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)

    # [V] + [V] + [V] -> [V, 3]
    return torch.stack([x, y, z], dim=-1)


def _least_squares_matrix(
    mics: torch.Tensor,
    freqs: torch.Tensor,
    speed_of_sound: float,
    directions: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    Build the frequency-dependent MIC-to-FOA encoding matrix
    by fitting the exact plane-wave response over many directions.

    Args:
        mics: [C, 3]
            C - number of microphones
            3 - microphone coordinates [x, y, z], in meters

        freqs: [F]
            F - number of STFT frequency bins
            Contains the frequency in Hz for each bin.

        speed_of_sound:
            Speed of sound in meters per second.

        directions: [V, 3]
            V - number of sampled directions
            3 - unit direction vector [x, y, z]

        beta:
            Ridge / Tikhonov regularization strength.

            At poorly determined frequencies, microphone responses can
            become very similar, making A @ A^H nearly singular or
            ill-conditioned. Adding beta * I keeps the system invertible
            and prevents excessively large encoding weights.

            Larger beta gives a more stable but more biased solution.

    Returns:
        E: [F, 4, C], complex
            F - number of frequency bins
            4 - FOA channels [W, X, Y, Z]
            C - number of microphone channels

            E[f] maps the C microphone spectra at frequency f
            to the four FOA channels.
    """
    C = mics.shape[0]
    F = freqs.shape[0]
    V = directions.shape[0]

    if directions.shape[-1] != 3:
        raise ValueError("directions must have shape [V, 3].")

    freqs = freqs.to(device=mics.device, dtype=mics.dtype)
    directions = directions.to(device=mics.device, dtype=mics.dtype)

    # array center as the FOA reference point
    D = mics - mics.mean(dim=0, keepdim=True) # [C, 3]

    # Spatial angular frequency (wavenumber):k = 2*pi*f / c
    k = 2 * torch.pi * freqs / speed_of_sound # [F]

    # project every microphone position onto every source direction
    projections = D @ directions.T # [C, 3] @ [3, V] = [C, V]

    # exact plane-wave steering matrix:
    # A[f, c, v] = exp(j * k[f] * <r_c, d_v>)
    #
    # A complex, the response of microphone c to a plane wave
    # arriving from direction v at frequency f
    phase = k[:, None, None] * projections[None, :, :]
    A = torch.exp(1j * phase)

    # ideal SN3D FOA response for every direction
    # W = 1, X = dx, Y = dy, Z = dz
    target = torch.cat(
        [
            torch.ones(
                (1, V),
                device=mics.device,
                dtype=mics.dtype,
            ),
            directions.T,
        ],
        dim=0,
    ).to(A.dtype) # [4, V]

    # Hermitian transpose:
    A_h = A.conj().transpose(-2, -1) # [F, C, V] -> [F, V, C]

    # Gram matrix A @ A^H
    # can become nearly singular / ill-conditioned
    # at frequencies where microphone responses are very similar
    gram = A @ A_h # [F, C, V] @ [F, V, C] = [F, C, C]

    # We solve: min_E ||E A - Y||_F^2 + beta ||E||_F^2
    # where:
    #       Y = target 
    #       beta = Ridge / Tikhonov regularization
    # which has the regularized solution: E = Y A^H (A A^H + beta I)^(-1)
    #
    # adding beta I shifts small eigenvalues away from zero,
    # stabilizing the solution when A A^H is nearly singular
    identity = torch.eye(
        C,
        device=mics.device,
        dtype=A.dtype,
    )

    regularized_gram = gram + beta * identity

    # Y @ A^H
    rhs = target.unsqueeze(0) @ A_h # [4, V] @ [F, V, C] = [F, 4, C]

    # we need E = rhs @ regularized_gram^(-1)
    # avoid explicitly forming the inverse, solve A @ X = B instead
    # torch.linalg.solve solves systems of the form .
    #
    # since regularized_gram is Hermitian:
    # E @ G = rhs
    # G @ E^H = rhs^H
    # E^H = solve(G, rhs^H)
    E_h = torch.linalg.solve(
        regularized_gram, # [F, C, C]
        rhs.mH, # F, C, 4]
    ) # [F, C, 4]

    # [F, C, 4] -> [F, 4, C]
    return E_h.mH


class FoaConverter:
    def __init__(
        self,
        config: StftConfig | None = None,
        mode: str = "gradient", # "gradient" | "ls"
        speed_of_sound: float = 343.0,
        max_gain_db: float = 20.0,  # gradient: caps low-freq noise boost
        rtol: float = 1e-6,  # gradient: rank cutoff for pinv
        n_directions: int = 642, # ls: grid size
        beta: float = 1e-3, # ls: regularisation
        regularize_gradient: bool = True,
        compensate_gradient: bool = True,
    ):
        if mode not in {"gradient", "ls"}:
            raise ValueError(
                f"mode must be 'gradient' or 'ls', got {mode!r}"
            )

        if speed_of_sound <= 0:
            raise ValueError("speed_of_sound must be positive.")

        if rtol < 0:
            raise ValueError("rtol must be non-negative.")

        if n_directions <= 0:
            raise ValueError("n_directions must be positive.")

        if beta < 0:
            raise ValueError("beta must be non-negative.")

        self.config = config if config is not None else StftConfig()

        self.mode = mode
        self.speed_of_sound = speed_of_sound

        self.max_gain_db = max_gain_db
        self.rtol = rtol
        self.regularize_gradient = regularize_gradient
        self.compensate_gradient = compensate_gradient

        self.n_directions = n_directions
        self.beta = beta

    def convert(self, 
                audio: torch.Tensor, # [N, C, T]
                mics: torch.Tensor, # [C, 3]
    ) -> torch.Tensor:
        """
        Convert multichannel microphone audio into First-Order Ambisonics
        (FOA) using the SN3D convention.

        Args:
            audio: [N, C, T]
                N - batch size
                C - number of microphone channels
                T - number of time-domain samples

            mics: [C, 3]
                C - number of microphones
                3 - microphone coordinates (x, y, z), in meters

        Returns:
            foa: [N, 4, T]
                N - batch size
                4 - FOA channels [W, X, Y, Z]
                T - number of time-domain samples

        The conversion is performed in the frequency domain:

            1. Compute the STFT of each microphone channel:
            [N, C, T] -> [N, C, F, T_frames]

            2. Build a frequency-dependent encoding matrix:
            E: [F, 4, C]

            For each frequency bin, E mixes the C microphone
            channels into the four FOA channels.

            3. Apply the encoding matrix independently at every
            batch item, frequency bin, and STFT time frame:
            [F, 4, C] x [N, C, F, T_frames]
            -> [N, 4, F, T_frames]

            4. Apply the inverse STFT to reconstruct the four
            time-domain FOA channels:
            [N, 4, F, T_frames] -> [N, 4, T]
        """
        if audio.ndim != 3:
            raise ValueError(f"audio must be [N, C, T], got shape {tuple(audio.shape)}")

        if mics.ndim != 2 or mics.shape[-1] != 3:
            raise ValueError(f"mics must be [C, 3], got shape {tuple(mics.shape)}")

        if audio.shape[1] != mics.shape[0]:
            raise ValueError(
                f"audio has {audio.shape[1]} channels but mics describes "
                f"{mics.shape[0]} microphones"
            )

        # the geometry is only ever used for arithmetic, so an integer tensor
        # of coordinates (an easy thing to type by hand) has to become float
        mics = mics.to(device=audio.device, dtype=audio.dtype)

        P = stft(audio, self.config)

        freqs = self.config.frequencies(
            device=audio.device,
            dtype=audio.dtype,
        )

        if self.mode == "gradient":
            E = _gradient_matrix(
                mics=mics,
                freqs=freqs,
                speed_of_sound=self.speed_of_sound,
                rtol=self.rtol,
                compensate=self.compensate_gradient,
                regularize=self.regularize_gradient,
                max_gain_db=self.max_gain_db,
            )
        else:
            directions = _fibonacci_sphere(
                n=self.n_directions,
                device=mics.device,
                dtype=mics.dtype,
            )

            E = _least_squares_matrix(
                mics=mics,
                freqs=freqs,
                speed_of_sound=self.speed_of_sound,
                directions=directions,
                beta=self.beta,
            )
        
        # mix C microphone channels into 4 FOA channels for each frequency bin
        # foa_spec[n, a, f, t] = nsum_c E[f, a, c] * P[n, c, f, t]
        foa_spec = torch.einsum(
            "fac,ncft->naft",
            E, # E: [F, 4, C]
            P, # P: [N, C, F, T_frames]
        ) # -> [N, 4, F, T_frames]

        # [N, 4, F, T_frames] -> [N, 4, T]
        return istft(
            foa_spec,
            self.config,
            length=audio.shape[-1],
        )
