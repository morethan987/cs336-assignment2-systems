import einx
import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root mean square normalization.
    State dict keys: weight
    Args:
        d_model (int): Hidden dimension of the model
        eps (float = 1e-5): Epsilon value for numerical stability
        device (torch.device | None = None): Device to store the parameters on
        dtype (torch.dtype | None=None): Data type of the parameters
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(self.d_model, **factory_kwargs))  # type: ignore
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rrms = torch.rsqrt(torch.divide(einx.dot("... in, ... in -> ...", x, x), self.d_model) + self.eps)
        res = einx.multiply("... in, in, ... -> ... in", x, self.weight, rrms)
        return res.to(in_dtype)


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = RMSNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1: torch.Tensor = self.fc1(x)
        print(f"Dtype of fc1 output: {out1.dtype}")
        x = self.relu(out1)
        x: torch.Tensor = self.ln(x)
        print(f"Dtype of ln output: {x.dtype}")
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = ToyModel(15, 15)
    model.to(device="cuda")
    input = torch.randn((4, 15), device="cuda")  # batch_size = 4
    target = torch.randint(0, 15, (4,), device="cuda")
    model.train()

    print("FP16 autocasting experiment")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        # type of model params
        for name, param in model.named_parameters():
            print(f"Param {name}, dtype {param.dtype}")
        y = model(input)
        print(f"Dtype of predicted logits: {y.dtype}")
        loss = nn.functional.cross_entropy(y, target)
        print(f"Dtype of loss: {loss.dtype}")

    # backward should follow the forward dtype
    loss.backward()
    print("Dtype of gradients:")
    for param in model.parameters():
        if param.grad is None:
            continue
        print(param.grad.dtype)

    print("=" * 50)
    for param in model.parameters():
        param.grad = None

    print("BF16 autocasting experiment")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(input)
        print(f"Dtype of predicted logits: {y.dtype}")
        loss = nn.functional.cross_entropy(y, target)
        print(f"Dtype of loss: {loss.dtype}")
    loss.backward()
    print("Dtype of gradients:")
    for param in model.parameters():
        if param.grad is None:
            continue
        print(param.grad.dtype)

    print("=" * 50)
    for param in model.parameters():
        param.grad = None

    print("RMSNorm sensitivity experiment")

    # Experiment 1: FP16 sensitivity to squaring (x^2) - Overflow Risk
    print("\n--- 1. FP16 Sensitivity: Squaring and Overflow ---")
    # Transformer layers often exhibit feature outliers reaching magnitudes of 200~300
    x_outlier = torch.tensor([300.0], device="cuda")

    sq_fp16 = (x_outlier.half()) ** 2
    sq_fp32 = (x_outlier.float()) ** 2

    print(f"Input feature value: {x_outlier.item()}")
    print(f"FP16 direct squaring (x^2): {sq_fp16.item()}  <-- (FP16 max is 65504, overflows to inf!)")
    print(f"Upcast to FP32 squaring (x^2): {sq_fp32.item()} <-- (Normal value)")

    # Experiment 2: BF16 sensitivity to summation (sum x^2) - Precision Loss
    print("\n--- 2. BF16 Sensitivity: Accumulation and Truncation Error (Precision Loss) ---")
    # Simulate a realistic hidden dimension d = 4096
    d = 4096
    x_large = torch.randn(d, device="cuda")

    # Use FP64 double precision as the ground truth baseline
    ref_variance = (x_large.double() ** 2).mean()

    # Pure BF16 accumulation vs. Upcasting to FP32 for accumulation
    var_pure_bf16 = (x_large.bfloat16() ** 2).mean()
    var_fp32 = (x_large.bfloat16().float() ** 2).mean()

    rel_err_bf16 = torch.abs(var_pure_bf16 - ref_variance) / ref_variance
    rel_err_fp32 = torch.abs(var_fp32 - ref_variance) / ref_variance

    print(f"Hidden Dimension d = {d}")
    print(f"FP64 Ground Truth variance:    {ref_variance.item():.6f}")
    print(f"Pure BF16 accumulation:        {var_pure_bf16.item():.6f} | Relative error: {rel_err_bf16.item():.4%}")
    print(f"Upcast to FP32 accumulation:   {var_fp32.item():.6f} | Relative error: {rel_err_fp32.item():.4%}")
    print("Conclusion: While BF16 avoids overflow, its 7-bit mantissa causes significant precision loss when summing over large dimensions.")

    # Experiment 3: End-to-end RMSNorm module verification with outliers
    print("\n--- 3. End-to-End RMSNorm Module Verification ---")
    norm_dim = 15
    rms = RMSNorm(norm_dim, device=torch.device("cuda"))

    # Create an input tensor injected with an activation outlier
    x_test = torch.randn((2, norm_dim), device="cuda")
    x_test[0, 0] = 300.0  # Inject outlier

    # Implementation A: Protected by internal upcast (x.to(torch.float32))
    out_safe = rms(x_test.half())

    # Implementation B: Naive pure FP16 execution without FP32 upcast
    x_h = x_test.half()
    mean_sq_fp16 = torch.divide(einx.dot("... in, ... in -> ...", x_h, x_h), norm_dim)
    rrms_fp16 = torch.rsqrt(mean_sq_fp16 + rms.eps)
    out_pure_fp16 = einx.multiply("... in, in, ... -> ... in", x_h, rms.weight.half(), rrms_fp16)

    print(f"Pure FP16 RMSNorm output contains NaN/Inf:   {torch.isnan(out_pure_fp16).any() or torch.isinf(out_pure_fp16).any()}")
    print(f"RMSNorm with FP32 cast output contains NaN/Inf: {torch.isnan(out_safe).any() or torch.isinf(out_safe).any()}")
