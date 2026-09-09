import torch
from cs336_basics.layers import TransformerLM, cross_entropy
from cs336_basics.train_loop import AdamW, ModelConfig, gradient_clipping
from torch.cuda import nvtx
from tqdm import tqdm

from cs336_systems.utils import BenchConfig, fake_cosine_annealing, parse_args


class NsysBenchMarker:
    def __init__(self, model_cfg: ModelConfig, bench_cfg: BenchConfig) -> None:
        self.model_cfg = model_cfg
        self.bench_cfg = bench_cfg

        # seeds and dirs
        self._set_seed()

        # construct model and optimizer
        self.model = TransformerLM(
            self.model_cfg.vocab_size,
            self.model_cfg.context_length,
            self.model_cfg.num_layers,
            self.model_cfg.d_model,
            self.model_cfg.num_heads,
            self.model_cfg.d_ff,
            self.model_cfg.rope_theta,
            self.model_cfg.device,
            self.model_cfg.dtype,
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.bench_cfg.lr,
            weight_decay=self.bench_cfg.weight_decay,
            eps=self.bench_cfg.eps,
            betas=self.bench_cfg.betas,
        )

    def _set_seed(self):
        torch.manual_seed(self.bench_cfg.torch_seed)

    def generate_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of input sequences and next-token targets.
        Returns:
            inputs: Tensor of shape (batch_size, context_length) on device.
            targets: Tensor of shape (batch_size, context_length) on device.
        """
        tks = torch.randint(
            0,
            self.model_cfg.vocab_size,
            (self.bench_cfg.batch_size, self.model_cfg.context_length + 1),
            device=self.model_cfg.device,
            dtype=torch.int,
        )
        return tks[:, :-1], tks[:, 1:]

    def _get_lr(self, step: int) -> float:
        return fake_cosine_annealing(
            t=step,
            alpha_max=self.bench_cfg.lr,
            alpha_min=self.bench_cfg.min_lr,
            t_w=self.bench_cfg.lr_warm_up,
            t_c=self.bench_cfg.steps,
        )

    # @nvtx.range("train_step")
    def train_step(self, step: int):
        # with nvtx.range("prepare"):
        lr_t = self._get_lr(step)
        self.optimizer.set_lr(lr_t)

        # take data
        x, targets = self.generate_data()

        # forward
        with nvtx.range("forward"):
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = cross_entropy(logits, targets)

        # backward
        # with nvtx.range("backward"):
        loss.backward()
        gradient_clipping(self.model.parameters(), max_l2_norm=self.bench_cfg.grad_clip)

        # optimizer
        # with nvtx.range("optimizer"):
        self.optimizer.step()

    def run(self):
        self.model.train()

        print(f"Steps for warm-up: {self.bench_cfg.warm_up}")
        for step in tqdm(range(1, self.bench_cfg.warm_up + 1), desc="warm-up", dynamic_ncols=True):
            self.train_step(step)
        torch.cuda.synchronize()

        print(f"Steps for benchmarking: {self.bench_cfg.steps}")
        torch.cuda.cudart().cudaProfilerStart()
        for step in tqdm(range(1, self.bench_cfg.steps + 1), desc="benchmarking", dynamic_ncols=True):
            self.train_step(step)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    mcfg, bcfg = parse_args()
    benchmarker = NsysBenchMarker(mcfg, bcfg)
    benchmarker.run()
