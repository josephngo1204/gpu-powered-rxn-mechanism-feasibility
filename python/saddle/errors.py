"""Exceptions shared by the GPU stack and the CPU-only pipeline.

Kept free of cupy/cuest imports so the workflow layer imports on machines
without a GPU (spec §14: a clean checkout passes all non-GPU tests).
"""


class SCFError(RuntimeError):
    pass


class PCMNotConverged(RuntimeError):
    def __init__(self, residual: float, iterations: int) -> None:
        super().__init__(
            f"PCM charges did not converge (residual {residual:.2e}, {iterations} iterations)"
        )
        self.residual = residual
        self.iterations = iterations
