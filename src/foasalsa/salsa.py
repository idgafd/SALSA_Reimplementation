import torch

from .stft import StftConfig, stft


def _covariance(spec: torch.Tensor, # [N, 4, F, T_frames]
                window: int,
    ) -> torch.Tensor: # [N, F, T_frames, 4, 4]
    """
    Estimate the spatial covariance matrix of every time-frequency bin.

    This is Eq. (5) of the paper (Section II.C). The true covariance
    R = E[X X^H] is unknowable from one recording, so it is approximated by
    averaging the instantaneous outer product over a few neighbouring frames,
    assuming sources move slowly inside that window.

    Args:
        spec: [N, 4, F, T_frames], complex
            N - batch size
            4 - FOA channels [W, X, Y, Z]
            F - number of frequency bins
            T_frames - number of STFT time frames

        window:
            Number of frames averaged together, the 2*Tr + 1 of Eq. (5).
            The paper uses Tr = 3, so a 7-frame window.

    Returns:
        cov: [N, F, T_frames, 4, 4], complex
            One Hermitian 4x4 matrix per time-frequency bin.

    A single frame gives R = X X^H, which has rank one by construction, so
    its second eigenvalue is zero and the coherence test of Section II.D
    would reject nothing. Averaging over several frames is what gives R a
    second eigenvalue to compare against, and so what makes the eigenvector
    approach work at all.
    """
    n_frames = spec.shape[-1]

    # cov[n, f, t, a, b] = X[n, a, f, t] * conj(X[n, b, f, t])
    outer = torch.einsum("naft,nbft->nftab", spec, spec.conj())

    if window <= 1:
        return outer

    half = window // 2

    # repeat the edge frames instead of padding with zeros, otherwise the
    # first and last few frames are averaged against silence and come out
    # much quieter than the rest
    index = torch.arange(
        -half,
        n_frames + half,
        device=spec.device,
    ).clamp(0, n_frames - 1)

    padded = outer[:, :, index]

    # moving average over the time axis, written the way Eq. (5) reads:
    # sum the window, then divide by its size
    total = padded[:, :, 0:n_frames]
    for shift in range(1, window):
        total = total + padded[:, :, shift : shift + n_frames]

    return total / window


def _power_iteration(cov: torch.Tensor, # [N, F, T_frames, 4, 4]
                     n_iterations: int,
                     eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find the principal eigenvector by repeated multiplication.

    Args:
        cov: [N, F, T_frames, 4, 4], complex
        n_iterations: How many times to multiply by cov.
        eps: Guard against dividing by a zero norm in silent bins.

    Returns:
        vector: [N, F, T_frames, 4], complex, unit norm
        value: [N, F, T_frames], the matching eigenvalue

    Convergence goes as (sigma2 / sigma1)^n_iterations, so it is slowest at
    bins where two sources are equally loud. Those bins fail the coherence
    test and are zeroed anyway.
    """
    # start from the first column of R. For a rank-one covariance that
    # column is already proportional to the steering vector, so this starts
    # from the classic intensity vector and refines it
    vector = cov[..., 0:1] # [N, F, T_frames, 4, 1]

    # a silent bin gives an all-zero column, which would divide by zero below
    fallback = torch.zeros_like(vector)
    fallback[..., 0, :] = 1.0
    vector = torch.where(
        vector.abs().sum(dim=-2, keepdim=True) < eps,
        fallback,
        vector,
    )

    for _ in range(n_iterations):
        vector = cov @ vector
        vector = vector / torch.linalg.vector_norm(
            vector, dim=-2, keepdim=True
        ).clamp(min=eps)

    # the vector is already unit norm so there is nothing to divide by
    # imaginary part is zero for a Hermitian matrix.
    value = (vector.mH @ cov @ vector).real[..., 0, 0]

    return vector[..., 0], value


def _principal_eigenvector(cov: torch.Tensor, # [N, F, T_frames, 4, 4]
                           method: str = "eigh",
                           n_power_iterations: int = 8,
                           eps: float = 1e-10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose every bin covariance into its dominant direction and a measure
    of how dominant it is.

    Args:
        cov: [N, F, T_frames, 4, 4], complex
            One Hermitian 4x4 matrix per time-frequency bin.

        method:
            "eigh"  - Hermitian eigendecomposition. The cheapest exact
                      method, because it uses the symmetry of the matrix.
            "svd"   - full singular value decomposition, which is what the
                      paper names in Eq. (10) for numerical stability. For a
                      Hermitian positive semi-definite matrix the two agree:
                      singular values are the eigenvalues.
            "power" - power iteration, approximate but batched matmuls only.

        n_power_iterations:
            Only used by "power".

        eps:
            Floor for divisions in empty bins.

    Returns:
        vector: [N, F, T_frames, 4], complex
            Principal eigenvector, unit norm. When one source dominates the
            bin, R is close to sigma^2 * H H^H and this vector approximates
            the steering vector H of Eq. (6).

        sigma1, sigma2: [N, F, T_frames]
            Two largest eigenvalues, largest first. Their ratio is the
            direct-to-reverberant ratio of Eq. (11).

    The covariance is Hermitian by construction, but floating point addition
    is not exactly symmetric, so the two triangles differ in the last digits.
    Both eigh and svd are given the symmetrised matrix. It costs one addition
    and means the result does not depend on which triangle a routine reads.
    """
    cov = 0.5 * (cov + cov.mH)

    if method == "eigh":
        # sorted ascending, the principal one is last
        values, vectors = torch.linalg.eigh(cov)

        return vectors[..., -1], values[..., -1], values[..., -2]

    if method == "svd":
        # for Hermitian PSD input the left singular vectors are the
        # eigenvectors and the singular values are the eigenvalues,
        # sorted descending
        vectors, values, _ = torch.linalg.svd(cov)

        return vectors[..., 0], values[..., 0], values[..., 1]

    if method == "power":
        vector, sigma1 = _power_iteration(cov, n_power_iterations, eps)

        # subtract the dominant component and repeat. The second eigenvalue
        # is then the largest one of what remains
        outer = vector.unsqueeze(-1) @ vector.unsqueeze(-1).mH
        deflated = cov - sigma1[..., None, None] * outer

        _, sigma2 = _power_iteration(deflated, n_power_iterations, eps)

        return vector, sigma1, sigma2

    raise ValueError(
        f"method must be 'eigh', 'svd' or 'power', got {method!r}"
    )


def _normalize_eigenvector(vector: torch.Tensor, # [N, F, T_frames, 4]
                           eps: float,
    ) -> torch.Tensor: # [N, F, T_frames, 3]
    """
    Turn a principal eigenvector into the EIV of Section II.C.1.

    Args:
        vector: [N, F, T_frames, 4], complex
            Principal eigenvector, defined only up to an arbitrary complex
            scale factor (the unknown loudness and phase of the source).

        eps:
            Floor for the two divisions below.

    Returns:
        eiv: [N, F, T_frames, 3]
            Real unit-norm direction vector per bin.

    The paper says to normalise by the first element, which is
    the omnidirectional channel, discard that element, take the real part,
    then normalise to unit length.

    A raw spatial pressure gradient is proportional to ±j*k*p*d
    (the sign depends on the Fourier convention). Thus XYZ are in quadrature 
    with W. Without the 1/(j*k) compensation performed by FoaConverter, 
    XYZ/W would be approximately imaginary, and taking its real part below 
    would destroy the directional information.
    """
    # dividing by the W element removes the unknown source amplitude and
    # phase, leaving the channel ratios that encode direction
    w = vector[..., 0:1]
    w = torch.where(w.abs() < eps, torch.full_like(w, eps), w)

    ratio = vector[..., 1:] / w

    # only the active part of the field points at the source
    # the reactive part carries no direction
    direction = ratio.real

    length = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)

    # a bin whose real part is zero has no direction to report, and
    # normalising it would turn rounding error into a unit vector
    return torch.where(
        length < eps,
        torch.zeros_like(direction),
        direction / length.clamp(min=eps),
    )


def _moving_average(values: torch.Tensor, # [N, F, T_frames]
                    window: int,
    ) -> torch.Tensor: # [N, F, T_frames]
    """Average over neighbouring frames, repeating the edges. Same as the
    smoothing in _covariance, but for a real-valued quantity."""
    if window <= 1:
        return values

    n_frames = values.shape[-1]
    half = window // 2

    index = torch.arange(
        -half,
        n_frames + half,
        device=values.device,
    ).clamp(0, n_frames - 1)

    padded = values[..., index]

    total = padded[..., 0:n_frames]
    for shift in range(1, window):
        total = total + padded[..., shift : shift + n_frames]

    return total / window


def _noise_floor(magnitude: torch.Tensor, # [N, F, T_frames]
                 rise: float,
                 fall: float,
                 n_init_frames: int,
                 eps: float,
    ) -> torch.Tensor: # [N, F, T_frames]
    """
    Track a slowly moving estimate of the background level per frequency.

    Args:
        magnitude: [N, F, T_frames]
            Magnitude of the reference channel, W for the FOA format.

        rise, fall:
            Multipliers applied when the current frame is above or below the
            running floor. Both are close to one, so the floor moves slowly
            instead of following the signal.

        n_init_frames:
            How many opening frames seed the floor. The paper assumes they
            contain noise only.

        eps:
            Keeps the floor from decaying all the way to zero in digital
            silence, where every later frame would then look like signal.

    Returns:
        floor: [N, F, T_frames]

    Section II.D describes this: "The noise floor is initialized using 
    the first few audio frames, which are assumed to contain only noise. 
    After that, the noise floor is slightly increased or decreased 
    if the magnitude of X1[t, f] is above or below the previous noise floor". 
    The multiplicative version below is the simplest one that matches that
    description. The paper notes that other estimators would also work.

    Note that this is the only sequential step in the package, because each
    frame depends on the previous one. The loop runs over frames, not over
    bins, so each iteration still works on a full [N, F] tensor.
    """
    n_frames = magnitude.shape[-1]

    floor = magnitude[..., :n_init_frames].mean(dim=-1).clamp(min=eps)

    history = []
    for t in range(n_frames):
        step = torch.where(magnitude[..., t] > floor, rise, fall)
        floor = (floor * step).clamp(min=eps)
        history.append(floor)

    return torch.stack(history, dim=-1)


def intensity_vector(cov: torch.Tensor, # [N, F, T_frames, 4, 4]
                     eps: float = 1e-10,
    ) -> torch.Tensor: # [N, F, T_frames, 3]
    """
    The classic intensity vector, for comparison against SALSA's EIV.

    Args:
        cov: [N, F, T_frames, 4, 4], complex
        eps: Floor for the divisions.

    Returns:
        iv: [N, F, T_frames], unit-norm direction per bin.

    This is *not* a SALSA feature. It is the LINSPEC_IV baseline of
    Section III.A, which the paper compares SALSA against, and it is here so
    the examples can show the difference on the same synthetic scene.

    The difference between the two is one line. Both come from the same
    covariance matrix: the baseline reads its first column, SALSA takes its
    principal eigenvector. The first column of R is

        R[1:, 0] = E[XYZ * conj(W)]

    whose real part is exactly the active intensity Re{W* [X, Y, Z]}.

    With one source and spatially white noise the two agree, because white
    noise only affects the W element. With two sources they differ: the first
    column returns sigma1^2 H1 + sigma2^2 H2, a weighted sum pointing at
    neither source, while the eigenvector follows the louder one.
    """
    return _normalize_eigenvector(cov[..., 0], eps)


class FoaSalsa:
    def __init__(
        self,
        config: StftConfig | None = None,
        method: str = "eigh", # "eigh" | "svd" | "power"
        cov_window: int = 7, # 2*Tr + 1 with Tr = 3, paper Eq. (5)
        fmin: float = 50.0, # paper V.A
        fmax: float = 9000.0, # paper V.A, set by the array's aliasing limit
        alpha_snr: float = 1.5, # paper Eq. (9)
        beta_drr: float = 5.0, # paper Eq. (11)
        apply_magnitude_test: bool = True,
        apply_coherence_test: bool = True,
        magnitude_window: int = 3, # running RMS width, paper Eq. (9)
        n_power_iterations: int = 8,
        noise_floor_rise: float = 1.02, # ours, the paper only says "slightly"
        noise_floor_fall: float = 0.98,
        n_init_frames: int = 5,
        eps: float = 1e-10,
    ):
        if method not in {"eigh", "svd", "power"}:
            raise ValueError(
                f"method must be 'eigh', 'svd' or 'power', got {method!r}"
            )

        if cov_window <= 0 or cov_window % 2 == 0:
            raise ValueError(
                f"cov_window must be a positive odd number, got {cov_window}"
            )

        if not 0 <= fmin < fmax:
            raise ValueError(f"need 0 <= fmin < fmax, got {fmin} and {fmax}")

        if alpha_snr < 0:
            raise ValueError("alpha_snr must be non-negative.")

        if beta_drr < 1:
            raise ValueError("beta_drr must be at least 1.")

        if n_power_iterations <= 0:
            raise ValueError("n_power_iterations must be positive.")

        if not 0 < noise_floor_fall <= 1 <= noise_floor_rise:
            raise ValueError(
                "need 0 < noise_floor_fall <= 1 <= noise_floor_rise, got "
                f"{noise_floor_fall} and {noise_floor_rise}"
            )

        if n_init_frames <= 0:
            raise ValueError("n_init_frames must be positive.")

        self.config = config if config is not None else StftConfig()

        self.method = method
        self.cov_window = cov_window

        self.fmin = fmin
        self.fmax = fmax

        self.alpha_snr = alpha_snr
        self.beta_drr = beta_drr
        self.apply_magnitude_test = apply_magnitude_test
        self.apply_coherence_test = apply_coherence_test
        self.magnitude_window = magnitude_window

        self.n_power_iterations = n_power_iterations
        self.noise_floor_rise = noise_floor_rise
        self.noise_floor_fall = noise_floor_fall
        self.n_init_frames = n_init_frames

        self.eps = eps

    def compute(self,
                foa: torch.Tensor, # [N, 4, T]
    ) -> torch.Tensor:
        """
        Computes SALSA features from a batch of FOA-encoded audio.

        Args:
            foa: [N, 4, T]
                N - batch size
                4 - FOA channels [W, X, Y, Z], SN3D
                T - number of time-domain samples

        Returns:
            features: [N, 7, F, T]
                N - batch size
                7 - four log-linear spectrograms plus three EIV channels
                F - number of frequency bins
                T - number of STFT time frames

                Note that the two T's above are not the same unit. The input
                counts audio samples at the sampling rate; the output counts
                STFT frames, which the paper's settings put at 80 per second
                (Section V.C). The task statement names both T, so the letter
                is kept.

        The seven channels are the two halves of Section II:

            0-3  log(|X|^2) for W, X, Y, Z, Eq. (2)
            4-6  the normalized principal eigenvector, Section II.C.1

        Both halves are on the same time-frequency grid, which is the point
        of the feature. Element (f, t) of channel 4 gives the direction of
        whatever produced the energy at element (f, t) of channel 0.
        Frame-level features such as GCC-PHAT cannot express this, because
        they have no frequency axis to attach a direction to.

        Steps:

            1. STFT of the four channels:
            [N, 4, T] -> [N, 4, F, T_frames]

            2. Log-linear spectrograms, Eq. (2):
            [N, 4, F, T_frames]

            3. Covariance of every bin over a few frames, Eq. (5):
            [N, F, T_frames, 4, 4]

            4. Principal eigenvector and the two leading eigenvalues:
            [N, F, T_frames, 4] and two of [N, F, T_frames]

            5. Normalize into the EIV, Section II.C.1:
            [N, F, T_frames, 3]

            6. Zero every bin that is not single-source, Section II.D,
            or that falls outside the [fmin, fmax] band.

            7. Stack both halves:
            -> [N, 7, F, T_frames]
        """
        if foa.ndim != 3:
            raise ValueError(f"foa must be [N, 4, T], got shape {tuple(foa.shape)}")

        if foa.shape[1] != 4:
            raise ValueError(
                f"foa must have 4 channels [W, X, Y, Z], got {foa.shape[1]}"
            )

        # [N, 4, T] -> [N, 4, F, T_frames]
        spec = stft(foa, self.config)

        # Eq. (2). The eps stops digital silence from becoming -inf
        linspec = torch.log(spec.abs().square() + self.eps)

        # [N, 4, F, T_frames] -> [N, F, T_frames, 4, 4]
        cov = _covariance(spec, self.cov_window)

        vector, sigma1, sigma2 = _principal_eigenvector(
            cov,
            method=self.method,
            n_power_iterations=self.n_power_iterations,
            eps=self.eps,
        )

        # [N, F, T_frames, 4] -> [N, F, T_frames, 3]
        eiv = _normalize_eigenvector(vector, self.eps)

        keep = self._single_source_bins(spec, sigma1, sigma2)

        # [N, F, T_frames] -> [N, F, T_frames, 1] so it zeroes whole vectors
        eiv = eiv * keep.unsqueeze(-1)

        # [N, F, T_frames, 3] -> [N, 3, F, T_frames]
        eiv = eiv.permute(0, 3, 1, 2)

        # [N, 4, F, T_frames] + [N, 3, F, T_frames] -> [N, 7, F, T_frames]
        return torch.cat([linspec, eiv.to(linspec.dtype)], dim=1)

    def _single_source_bins(self,
                            spec: torch.Tensor, # [N, 4, F, T_frames]
                            sigma1: torch.Tensor, # [N, F, T_frames]
                            sigma2: torch.Tensor, # [N, F, T_frames]
    ) -> torch.Tensor: # [N, F, T_frames]
        """
        Decide which bins carry a usable direction, Section II.D.

        Returns a float mask rather than a boolean one so it can be
        multiplied straight into the EIV.

        The two tests reject different things. The magnitude test drops bins
        with no foreground sound in them. The coherence test drops bins that
        do contain sound, but from more than one direction, or from one
        direction plus a lot of reverberation. A bin must pass both.

        This split keeps direction and confidence separate. The EIV carries
        only direction, at unit length, and the mask carries only confidence,
        as zero or one. An energy-normalised intensity vector puts both into
        one magnitude, where a short vector cannot be told apart from a
        confident vector pointing between two sources.
        """
        keep = torch.ones_like(sigma1)

        if self.apply_magnitude_test:
            # X1 in Eq. (9) is the reference channel, W for the FOA format
            magnitude = spec[:, 0].abs()

            # "a running root-mean-square of the magnitude of X1 over a
            # 3-frame window", Eq. (9)
            smoothed = _moving_average(
                magnitude.square(), self.magnitude_window
            ).sqrt()

            floor = _noise_floor(
                magnitude,
                rise=self.noise_floor_rise,
                fall=self.noise_floor_fall,
                n_init_frames=self.n_init_frames,
                eps=self.eps,
            )

            keep = keep * (smoothed > self.alpha_snr * floor)

        if self.apply_coherence_test:
            # Eq. (11). Eigenvalues of a positive semi-definite matrix are
            # never negative in exact arithmetic, but rounding can push them
            # slightly below zero, and a negative sigma2 would flip the
            # comparison
            sigma1 = sigma1.clamp(min=0.0)
            sigma2 = sigma2.clamp(min=0.0)

            # sigma2 near zero means a rank-one bin, which is the
            # single-source case, so a very large ratio should pass
            keep = keep * (sigma1 > self.beta_drr * sigma2)

        # the band limits of Section V.A. Below fmin there is no useful
        # signal, above fmax the array is spatially aliased
        freqs = self.config.frequencies(device=keep.device, dtype=keep.dtype)
        in_band = (freqs >= self.fmin) & (freqs <= self.fmax)

        return keep * in_band[None, :, None]
