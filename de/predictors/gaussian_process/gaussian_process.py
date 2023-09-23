from typing import List
# from strkernel.mismatch_kernel import preprocess, MismatchKernel
from .kernels.mismatch import preprocess, MismatchKernel
from ._gpr import GaussianProcessRegressor


class GaussianProcess:
    def __init__(self, k: int, alphabet_size: int = 20, max_num_mismatch: int = 0):
        self.alphabet_size = alphabet_size
        self.k = k
        self.m = max_num_mismatch
        self.gpr = GaussianProcessRegressor(kernel="precomputed")
        # self.kernel = MismatchKernel(l=alphabet_size, k=k, m=max_num_mismatch)
        self.train_seqs = None

    def compute_kernel(self, seqs: List[str]):
        kernel = MismatchKernel(l=self.alphabet_size, k=self.k, m=self.m)
        processedX = preprocess(seqs)
        processedX = kernel.get_kernel(processedX).kernel
        return processedX

    def fit(self, seqs: List[str], y):
        processedX = self.compute_kernel(seqs)
        self.gpr.fit(processedX, y)
        self.train_seqs = seqs

    def score_mutants(self, seqs: List[str]):
        total_seqs = seqs + self.train_seqs
        processedX = self.compute_kernel(total_seqs)
        inputs = processedX[:len(seqs), len(seqs):]
        score = self.gpr.predict(inputs)
        return score
