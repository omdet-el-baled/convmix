from .samplers import initialize_terminal_state


class RepeatedObservationInitialization:
    def __call__(self, sde, y, K, n_fft, add_noise, *, noise_source):
        return initialize_terminal_state(sde, y, K, n_fft, add_noise, noise_source=noise_source)
