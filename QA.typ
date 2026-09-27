#import "@preview/ilm:2.1.1": *
#import "@preview/subpar:0.2.2"

#show: ilm.with(
  title: [CS336 Assignment 2],
  authors: "Morethan",
  date: datetime(year: 2026, month: 09, day: 04),
  abstract: [Language Modeling from Scratch - Systems and Parallelism],
  bibliography: none,
  figure-index: (enabled: false),
  table-index: (enabled: false),
  listing-index: (enabled: false),
)
#set text(lang: "en")
#set page(paper: "a4", margin: (x: 1cm, y: 1cm))
#set enum(numbering: "(a)")

#let response(title: "Response", body) = block(
  stroke: 0.5pt + luma(180),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  breakable: true,
)[
  #if title != none [*#title:*\ ]
  #body
]

/////////////////////////////////////////////

= Profiling and Benchmarking

== Preliminary

#figure(
  caption: "Specifications of different model sizes.",
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: (x: 8pt, y: 4.5pt),
    align: (left, center, center, center, center),
    stroke: none,

    table.hline(stroke: 1.2pt),
    [Size], [`d_model`], [`d_ff`], [`num_layers`], [`num_heads`],
    table.hline(stroke: 0.6pt),

    [small], [768], [3072], [12], [12],
    [medium], [1024], [4096], [24], [16],
    [large], [1280], [5120], [36], [20],
    [xl], [2560], [10240], [32], [32],
    [10B], [4608], [12288], [50], [36],
    table.hline(stroke: 1.2pt),
  ),
)<model-specs>

Context length is 512 unless otherwise specified.

== Benchmarking Script

+ Write a script to perform basic end-to-end benchmarking of the forward pass, backward pass, and optimizer step in your model. Specifically, your script should support the following:
  - Given hyperparameters (e.g., number of layers), initialize a model.
  - Generate a random batch of data.
  - Run $w$ warm-up steps (before you start measuring time), then time the execution of $n$ steps (either only forward, forward and backward, or forward and backward with optimizer step, depending on an argument). For timing, you can use the Python `timeit` module (e.g., either using the `timeit` function, or using `timeit.default_timer()`, which gives you the system's highest resolution clock, thus a better default for benchmarking than `time.time()`).
  - Call `torch.cuda.synchronize()` after each step.

  *Deliverable:* A script that will initialize a `basics` Transformer model with the given hyperparameters, create a random batch of data, and time forward-only, forward-and-backward, and full training steps that include the optimizer step.

  #response[See `benchmark_script.py`]

+ Time the forward, backward, and optimizer step for the model sizes described in Section 2.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?

  *Deliverable:* A 1-2 sentence response with your timings.

  #response[
    I measured the timing with various steps and 5 warm-up steps. A forward pass takes 40.33 ms with 2.71 ms standard deviation (relatively 6.72%), a backward pass takes 43.53 ms with 4.02 ms standard deviation (relatively 9.23%). Besides, more evaluated steps do not make markedly difference.
    #figure(
      table(
        columns: (auto, auto, auto, auto, auto, auto, auto),
        inset: (x: 8pt, y: 4.5pt),
        align: (left, right, right, right, right, right, right),
        stroke: none,

        // top line
        table.hline(stroke: 1.2pt),
        table.header(
          table.cell(rowspan: 2, align: horizon + left)[*Stage*],
          table.cell(colspan: 2, align: center)[*50 Steps (ms)*],
          table.cell(colspan: 2, align: center)[*150 Steps (ms)*],
          table.cell(colspan: 2, align: center)[*500 Steps (ms)*],
          table.hline(start: 1, end: 7, stroke: 0.5pt),
          [Mean], [Std], [Mean], [Std], [Mean], [Std],
        ),

        // header split
        table.hline(stroke: 0.6pt),
        [Prepare], [0.0793], [0.0076], [0.0813], [0.0090], [0.0806], [0.0223],
        [Forward], [40.3289], [2.7126], [42.4276], [3.2337], [41.5421], [2.2385],
        [Backward], [43.5285], [4.0191], [44.4743], [2.2116], [44.0380], [3.1208],
        [Optimizer], [5.6423], [0.9954], [5.4675], [0.4122], [5.5833], [0.6507],

        table.hline(stroke: 0.4pt),
        [Total], [89.5790], [5.5102], [92.4508], [4.5914], [91.2441], [4.4689],

        // bottom
        table.hline(stroke: 1.2pt),
      ),
      caption: [Benchmarking with 5 warm-up steps and 50 evalued steps on `small` model with 512 context length],
    )
  ]

+ One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?

  *Deliverable:* A 2-3 sentence response.

  #response[
    Without warm-up steps, the mean latency at each stage increases markedly, and the total standard deviation is over 17 times higher than that with warm-up. This discrepancy is primarily attributed to the cold-start overheads of the GPU runtime and deep learning framework—such as CUDA context initialization, PyTorch memory caching allocator initialization (avoiding repeated `cudaMalloc`). Incorporating just 1 or 2 warm-up steps effectively eliminates these initialization artifacts, resulting in stable performance with significantly reduced standard deviation.

    *Note: The latency is not driven by GPU hardware cache which is limited to tens of megabytes. It is completely flushed during a single forward and backward pass.*

    #figure(
      table(
        columns: (auto, auto, auto, auto, auto, auto, auto),
        inset: (x: 8pt, y: 4.5pt),
        align: (left, right, right, right, right, right, right),
        stroke: none,

        // top line
        table.hline(stroke: 1.2pt),
        table.header(
          table.cell(rowspan: 2, align: horizon + left)[*Stage*],
          table.cell(colspan: 2, align: center)[*0 warm-up (ms)*],
          table.cell(colspan: 2, align: center)[*1 warm-up (ms)*],
          table.cell(colspan: 2, align: center)[*2 warm-up (ms)*],
          table.hline(start: 1, end: 7, stroke: 0.5pt),
          [Mean], [Std], [Mean], [Std], [Mean], [Std],
        ),

        // header split
        table.hline(stroke: 0.6pt),
        [Prepare], [0.1686], [0.6106], [0.0754], [0.0097], [0.0781], [0.0107],
        [Forward], [59.1047], [130.3580], [39.3797], [4.0224], [39.8591], [3.9584],
        [Backward], [44.4363], [14.5282], [41.8902], [4.3877], [42.6913], [4.5618],
        [Optimizer], [5.8531], [2.2878], [5.4175], [0.6076], [5.4019], [0.4402],

        table.hline(stroke: 0.4pt),
        [Total], [109.5627], [147.1171], [86.7629], [8.6414], [88.0305], [8.3879],

        // bottom
        table.hline(stroke: 1.2pt),
      ),
      caption: [Warm-up ablation with 50 evaluated steps on `small` model with 512 context length],
    )
  ]

== Nsight Systems Profiling

Profile your forward pass, backward pass, and optimizer step using nsys with two model sizes from Table 1 of your choice as well as three power-of-two context lengths larger than 128, where the largest available size should be the longest context length you can fit in memory. Pick the combinations you think would be the most interesting to look at. For each profile answer the following questions:

+ What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?

  *Deliverable*: A 1-2 sentence response.

  #response[
    Forward time costs varies between models. The time cost of `small_ctx512` is 5 ms larger than that measured by naive method which is probably caused by `nsys`.

    #figure(
      table(
        columns: (auto, auto, auto, auto, auto, auto),
        inset: (x: 8pt, y: 4.5pt),
        align: (left, right, right, right, right, right),
        stroke: none,

        // top line
        table.hline(stroke: 1.2pt),
        table.header(
          table.cell(rowspan: 2, align: horizon + left)[*Stage*],
          table.cell(colspan: 3, align: center)[*Small*],
          table.cell(colspan: 2, align: center)[*Medium*],
          table.hline(start: 1, end: 6, stroke: 0.5pt),
          [ctx512], [ctx1024], [ctx2048], [ctx512], [ctx1024],
        ),

        // header split
        table.hline(stroke: 0.6pt),
        [Prepare], [0.0013], [0.0016], [0.0016], [0.0014], [0.0014],
        [Forward], [45.8780], [35.3587], [114.0988], [71.3086], [99.9471],
        [Backward], [47.5961], [83.2663], [257.1673], [77.7094], [219.9908],
        [Optimizer], [7.6235], [3.3185], [3.3330], [9.7619], [10.3211],

        // bottom
        table.hline(stroke: 1.2pt),
      ),
      caption: [Benchmarking with `nsys`],
    )
  ]

+ What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes? (Hint: look at the “CUDA GPU Kernel Summary” under “Stats System View”, and filter using NVTX ranges to identify which parts of the model are responsible for which kernels.)

  *Deliverable*: A 1-2 sentence response.

  #response[
    It is `gemm`, which means General Matrix Multiply ($D = alpha A B + beta C$), that dominants the forward pass time cost. The only exception occures on `small_ctx2048` model where the `masked_fill` is the most expensive. This should be attributed to computation and memory grap -- computation has a much larger throughput than memory operations.

    The `gemm` is called at lease 25 times during a single forward pass. The most expensive kernel is the same kind when input is small (both are `gemm` or its fused version). But the `vectorized_mul` surpass `gemm` during backward pass when input is large due to the heavy memory movement when apllying elementwise multiplication.

    By the way, the size of `gemm` is not the size of input data because of the auto-tiling in GPU.

    #figure(
      table(
        columns: (auto, auto, auto, auto, auto, auto),
        inset: (x: 6pt, y: 4.5pt),
        align: (left, center, center, center, center, center),
        stroke: none,

        // top line
        table.hline(stroke: 1.2pt),
        table.header(
          table.cell(rowspan: 2, align: horizon + left)[*Stage*],
          table.cell(colspan: 3, align: center)[*Small*],
          table.cell(colspan: 2, align: center)[*Medium*],
          table.hline(start: 1, end: 6, stroke: 0.5pt),
          [ctx512], [ctx1024], [ctx2048], [ctx512], [ctx1024],
        ),

        // header split
        table.hline(stroke: 0.6pt),
        [Kernel], [gemm(128×128)], [gemm(128×64)], [masked_fill], [gemm(128×128)], [gemm(64×128)],
        [cumu_time], [9.1304], [17.2556], [55.7434], [55.2499], [58.8550],
        [count], [25], [60], [12], [168], [48],
        [avg_time], [0.0730], [0.0575], [0.9291], [0.0658], [0.2452],

        // bottom
        table.hline(stroke: 1.2pt),
      ),
      caption: [Forward Top Kernel],
    )
    #figure(
      table(
        columns: (auto, auto, auto, auto, auto, auto),
        inset: (x: 6pt, y: 4.5pt),
        align: (left, center, center, center, center, center),
        stroke: none,

        // top line
        table.hline(stroke: 1.2pt),
        table.header(
          table.cell(rowspan: 2, align: horizon + left)[*Stage*],
          table.cell(colspan: 3, align: center)[*Small*],
          table.cell(colspan: 2, align: center)[*Medium*],
          table.hline(start: 1, end: 6, stroke: 0.5pt),
          [ctx512], [ctx1024], [ctx2048], [ctx512], [ctx1024],
        ),

        // header split
        table.hline(stroke: 0.6pt),
        [Kernel], [gemm_relu], [vectorized_mul], [vectorized_mul], [gemm(128×128)], [vectorized_mul],
        [cumu_time], [24.0571], [56.7776], [192.7157], [54.0182], [152.5319],
        [count], [109], [72], [72], [168], [144],
        [avg_time], [0.0441], [0.1577], [0.5353], [0.0643], [0.2118],

        // bottom
        table.hline(stroke: 1.2pt),
      ),
      caption: [Backward Top Kernel],
    )
  ]

+ Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?

  *Deliverable*: A 1-2 sentence response.

  #response[
    The other kernels that account non-trivical runtime are elementwise operations (e.g., vectorized mul, scalar mul, elementwise addition or exp) or memory access operations (e.g., masked fill, tensor copy)
  ]

+ Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you'd do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?

  *Deliverable*: A 1-2 sentence response.

  #response[
    The feaction of time spent on matmul decreases during a complete training step compared to forward-only inference. Conversely, other kernels (such as element-wise operations) account for a larger proportion of the overall runtime.

    #figure(
      table(
        columns: (auto, auto, auto, auto, auto, auto),
        inset: (x: 6pt, y: 4.5pt),
        align: (left, center, center, center, center, center),
        stroke: none,

        // top line
        table.hline(stroke: 1.2pt),
        table.header(
          table.cell(rowspan: 2, align: horizon + left)[*Stage*],
          table.cell(colspan: 3, align: center)[*Small*],
          table.cell(colspan: 2, align: center)[*Medium*],
          table.hline(start: 1, end: 6, stroke: 0.5pt),
          [ctx512], [ctx1024], [ctx2048], [ctx512], [ctx1024],
        ),

        // header split
        table.hline(stroke: 0.6pt),
        [Forward only], [47.0%], [31.7%], [24.8%], [50.4%], [34.9%],
        [Train step], [40.8%], [30.6%], [24.5%], [41.4%], [32.7%],

        // bottom
        table.hline(stroke: 1.2pt),
      ),
      caption: [Matmul ratios on different ranges],
    )
  ]

+ Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

  *Deliverable*: A 1-2 sentence response.

  #response[
    The matmul operation takes 102.40 times more FLOPs but cost less half of the time than that of softmax operation. This finally leads to hundreds times grap on throughput. This is the strongest motivation for FlashAttention.
    #let x = math.times
    #figure(
      table(
        columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
        inset: (x: 5pt, y: 4.5pt),
        align: (left, center, center, center, center, center, center, center, center),
        stroke: none,

        table.hline(stroke: 1.2pt),
        table.header(
          table.cell(rowspan: 2, align: horizon + left)[*Config*],
          table.cell(colspan: 3, align: center)[*Matmul($Q K^T + A V$)*],
          table.cell(colspan: 3, align: center)[*Softmax*],
          table.cell(colspan: 2, align: center)[*Ratio(MM/SM)*],
          table.hline(start: 1, end: 9, stroke: 0.5pt),
          [Time (ms)], [GFLOPs], [TFLOP/s], [Time (ms)], [GFLOPs], [TFLOP/s], [Time], [Throughput],
        ),

        table.hline(stroke: 0.6pt),
        [`small_ctx512`], [2.82], [965.5], [342.38], [5.91], [9.4], [1.60], [0.48#x], [214.4#x],
        [`small_ctx1024`], [15.24], [3865.2], [253.62], [51.00], [37.8], [0.74], [0.30#x], [342.7#x],
        [`small_ctx2048`], [58.51], [15460.7], [264.24], [212.49], [151.0], [0.71], [0.28#x], [371.9#x],
        table.hline(stroke: 0.3pt),
        [`medium_ctx512`], [6.76], [2578.7], [381.46], [16.29], [25.2], [1.54], [0.41#x], [246.9#x],
        [`medium_ctx1024`], [40.52], [10309.1], [254.42], [142.81], [100.7], [0.70], [0.28#x], [360.9#x],

        table.hline(stroke: 1.2pt),
      ),
      caption: [
        Runtime, FLOPs, and Throughput comparison between MatMul ($Q K^T$ and $A V$) and Softmax within the Self-Attention forward pass. Note that the theoretical FLOPs ratio is constant at $102.40times$.
      ],
    )
  ]

== Mixed precision accumulation

Run the following code and comment on the accuracy of the results.

```py
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
  s += torch.tensor(0.01,dtype=torch.float32)
print(s)

s = torch.tensor(0,dtype=torch.float16)
for i in range(1000):
  s += torch.tensor(0.01,dtype=torch.float16)
print(s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
  s += torch.tensor(0.01,dtype=torch.float16)
print(s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
  x = torch.tensor(0.01,dtype=torch.float16)
  s += x.type(torch.float32)
print(s)
```

*Deliverable*: A 2-3 sentence response.

#response[
  Pure FP16 yields the worst accuracy, with an error 496 times that of pure FP32. The last two results indicate that automatic type casting occurs during the addition.
  ```txt
  tensor(10.0001)
  tensor(9.9531, dtype=torch.float16)
  tensor(10.0021)
  tensor(10.0021)
  ```
]

== Benchmarking mixed precision

+ Consider the following model:

  ```py
  class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
      super().__init__()
      self.fc1 = nn.Linear(in_features, 10, bias=False)
      self.ln = nn.LayerNorm(10)
      self.fc2 = nn.Linear(10, out_features, bias=False)
      self.relu = nn.ReLU()

    def forward(self, x):
      x = self.relu(self.fc1(x))
      x = self.ln(x)
      x = self.fc2(x)
      return x
  ```

  Suppose we are training the model on a GPU and that the model parameters are originally in FP32. We'd like to use autocasting mixed precision with FP16. What are the data types of:

  - the model parameters within the autocast context?
  - the output of the first feed-forward layer (`ToyModel.fc1`)?
  - the output of layer norm (`ToyModel.ln`)?
  - the model's predicted logits?
  - the loss?
  - the model's gradients?

  *Deliverable*: The data types for each of the components listed above.

  #response[
    Model parameters keeps FP32 since the autocasting only applied when operations are launched to GPU. `ToyModel.fc1` yields FP16, `ToyModel.ln` yields FP32. Model's predicted logits is FP16. The loss is FP32. The gradients are all FP32.
  ]

+ You should have seen that FP16 mixed precision autocasting treats the layer normalization layer differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently? Why or why not?

  *Deliverable*: A 2-3 sentence response.

  #response[
    The squaring operation is sensitive to FP16 for overflowing. The accumulation sum is sensitive to BF16 due to truncation error. Actually, PyTorch does cast layer normalization to FP32 even using BF16. Though the error caused by BF16 is relatively small than FP16, there is still a notable error when applying reduction on large dimensions.

    ```txt
    --- 1. FP16 Sensitivity: Squaring and Overflow ---
    Input feature value: 300.0
    FP16 direct squaring (x^2): inf  <-- (FP16 max is 65504, overflows to inf!)
    Upcast to FP32 squaring (x^2): 90000.0 <-- (Normal value)

    --- 2. BF16 Sensitivity: Accumulation and Truncation Error (Precision Loss) ---
    Hidden Dimension d = 4096
    FP64 Ground Truth variance:    1.025929
    Pure BF16 accumulation:        1.023438 | Relative error: 0.2429%
    Upcast to FP32 accumulation:   1.025927 | Relative error: 0.0002%
    ```
  ]

+ Modify your benchmarking script to optionally run the model using mixed precision with BF16. Time the forward and backward passes with and without mixed-precision for each language model size described in @model-specs. Compare the results of using full precision versus mixed precision, and comment on any trends as model size changes. You may find the nullcontext no-op context manager to be useful.

  *Deliverable*: A 2-3 sentence response with your timings and commentary.

  #response[
    An RTX 4090 can only handle up to the `medium`-sized model, but it is sufficient to draw some key conclusions. First, autocasting is not free and actually increases forward pass latency, which negatively impacts the small model. The real benefit of autocasting comes from the backward pass and larger model sizes. Backward pass and larger model contain more FLOPs and heavier memory access which is signigicantly affected by data precision.

    #figure(
      table(
        columns: (auto, auto, auto, auto, auto),
        inset: (x: 8pt, y: 5pt),
        align: (left, center, center, center, center),
        stroke: none,

        // top line
        table.hline(stroke: 1.2pt),
        table.header(
          table.cell(rowspan: 2, align: horizon + left)[*Stage*],
          table.cell(colspan: 2, align: center)[*Small (ms)*],
          table.cell(colspan: 2, align: center)[*Medium (ms)*],
          table.hline(start: 1, end: 5, stroke: 0.5pt),
          [Full], [Mixed], [Full], [Mixed],
        ),

        // header split
        table.hline(stroke: 0.6pt),
        [Forward], [29.85 ± 3.32], [41.83 ± 6.38], [84.58 ± 0.16], [67.47 ± 12.85],
        [Backward], [63.07 ± 0.18], [45.69 ± 2.77], [179.06 ± 0.35], [100.52 ± 0.80],
        table.hline(stroke: 0.3pt),
        [*Total*], [*98.82 ± 3.35*], [*93.51 ± 8.16*], [*282.13 ± 0.46*], [*186.60 ± 13.27*],

        // bottom
        table.hline(stroke: 1.2pt),
      ),
      caption: [Latency comparison between full and mixed precision],
    )
  ]

== Memory profiling

Profile your complete training step of forward pass, backward pass, and optimizer step of the `xl` model from @model-specs with context lengths of 128 and 2048.

_*Length of 2048 on `xl` model is out of the capacity of RTX PRO 6000 with 84 GiB cache which is the best GPU I could rent. The experiment available is `large` model with 128, 512 and 1024 context length, `xl` model with 128 and 512 context length.*_

+ Add an option to your profiling script to run your model through the memory profiler. It may be helpful to reuse some of your previous infrastructure (e.g., to activate mixed-precision, load specific model sizes, etc). Then, run your script to get a memory profile of the `xl` model when either doing inference only (just forward pass) or a full training step. What do your memory timelines look like? Can you tell which stage is running based on the peaks you see?

  *Deliverable*: Two images of the "Active memory timeline" of an `xl` model, from the `memory_viz` tool: one for the forward pass, and one for running a full training step (forward and backward passes, then optimizer step), and a 2-3 sentence response.

  #response[
    The timeline for inference looks like a solid rectangle with spikes on the top and that for training is like 5 pyramid with each correspondes to a complete training step.The spikes in inference mode are made by the attention layer so there are 160 spikes (5 steps and each step has 32 attention layers). In training mode, the increasing side is the forward pass, the decreasing side is the backward pass. The tiny plateau following the backward pass is the period before the next `optimizer.zero_grad()`. The hight of the plateau is the memory size of gradients.

    #subpar.grid(
      columns: (1fr, 1fr),
      gutter: 1em,
      figure(
        image("assets/memory_xl_infer_w5_s5_ctx1024.png"),
        caption: [Inference mode],
      ),
      <fig:memory-infer>,

      figure(
        image("assets/memory_xl_train_w5_s5_ctx1024.png"),
        caption: [Training mode],
      ),
      <fig:memory-train>,

      caption: [Memory profiling on `xl` model with 5 warm-up steps, 5 evaluated steps, and a context length of 512.],
      label: <fig:memory-xl>,
    )
  ]

+ What is the peak memory usage of each context length when doing a forward pass? What about when doing a full training step?

  *Deliverable*: A table with two numbers per context length.

  #response[
    The data is generated by `uv run cs336_systems/memory_pickle_analyse/parse_mem.py ./memory_res`.

    #figure(
      table(
        columns: (auto, auto, auto, auto, auto, auto),
        inset: (x: 8pt, y: 5.5pt),
        align: (center + horizon, center + horizon, center, center, center, center),
        stroke: none,

        table.hline(stroke: 1.2pt),
        table.header([*Model*], [*Task*], [*Context*], [*FP32(GiB)*], [*Mixed BF16(GiB)*], [*Reduction*]),
        table.hline(stroke: 0.6pt),

        table.cell(rowspan: 6)[*Large*],
        table.cell(rowspan: 3)[Infer],
        [128], [3.78], [5.56], [+47.09%],
        [512], [4.01], [5.74], [+43.14%],
        [1024], [4.77], [6.48], [+35.85%],
        table.hline(start: 1, stroke: 0.3pt),
        table.cell(rowspan: 3)[Train],
        [128], [14.94], [15.21], [+1.81%],
        [512], [27.53], [24.55], [-10.82%],
        [1024], [55.59], [45.35], [-18.42%],

        table.hline(stroke: 0.8pt),

        table.cell(rowspan: 4)[*XL*],
        table.cell(rowspan: 2)[Infer],
        [128], [12.93], [19.18], [+48.34%],
        [512], [13.30], [19.38], [+45.71%],
        table.hline(start: 1, stroke: 0.3pt),
        table.cell(rowspan: 2)[Train],
        [128], [51.13], [51.10], [-0.06%],
        [512], [65.50], [63.90], [-2.44%],

        table.hline(stroke: 1.2pt),
      ),
      caption: [Peak memory usage comparison with and without mixed precision.],
    ) <peak_memory>
  ]

+ Find the peak memory usage of the `xl` model when using mixed-precision, for both a forward pass and a full training step. Does mixed-precision significantly affect memory usage?

  *Deliverable*: A 2-3 sentence response.

  #response[
    For the `xl` model with mixed-precision, the peak forward pass memory is 19.18 GiB (ctx=128) and 19.38 GiB (ctx=512), while a full training step reaches 51.10 GiB (ctx=128) and 63.90 GiB (ctx=512). Mixed-precision significantly affects inference by increasing memory by 46–48% due to PyTorch dynamically caching an additional BF16 copy of the model weights. However, its effect during training is negligible for shorter contexts (-0.06% to -2.44%) because the large static memory footprint of optimizer states and FP32 master weights heavily dominates over activation savings.
  ]

+ Consider the `xl` model. Given our reference hyperparameters, what is the size of a tensor of activations in the Transformer residual stream, in single-precision? Give this size in MiB (i.e., divide the number of bytes by $1024^2$ ).

  *Deliverable*: A 1-2 sentence response with your derivation.

  #response[
    The size of residual stream tensor is always the same that is (batch_size, sequence_length, d_model). So the activation size is 80 MiB.
  ]

+ Now look closely at the "Active Memory Timeline" from #link("https://docs.pytorch.org/memory_viz")[pytorch.org/memory_viz] of a memory snapshot of the `xl` model doing a forward pass. When you reduce the "Detail" level, the tool hides the smallest allocations to the corresponding level (e.g., putting "Detail" at 10% only shows the 10% largest allocations). What is the size of the largest allocations shown? Looking through the stack trace, can you tell where those allocations come from?

  *Deliverable*: A 1-2 sentence response.

  #response[
    The largest size is 128.0 MiB. These allocations come from the `scaled_dot_product_attention` function written in assignment 1.
  ]

+ Nsight Systems also has flags for memory profiling. You can combine these with the Nsight flags from before to understand what allocations are happening at different steps in your model's lifespan. Use the PyTorch-provided NVTX labels to determine how much memory is saved for backward (these tensors are often called residuals) by a single `TransformerBlock` in your model. Note the 5 largest contributing operations, and what percentage of the overall memory they contribute.

  During the backward pass, all these tensors will be freed, but new gradient tensors are emitted at the same time. Based on your profiles showing how much memory was allocated during the forward pass, and how much memory usage changes for every `TransformerBlock` in the backward pass, calculate how much memory the produced gradient tensors for a `TransformerBlock` take. Does the result match what you expect?

  *Deliverable*: Screenshots from Nsight Systems and a 1-2 paragraph response.

  #response[
    Nsight System get some errors when plotting memory usage figures. So I write some customized script to analyse sqlite file exported by `nsys` and yields a reconstructed timeline. Get detailed analysis report by running `uv run cs336_systems/nsys_analyse/analyze_memory.py profiles/some_experiment`.

    #figure(
      caption: [Nsight System Memory error.],
      image("assets/nsys_memory.png"),
    )

    For a single `TransformerBlock` (e.g., `TransformerLM.transformers.10`), the forward pass allocates a total of 1671.07 MiB and temporarily frees 813.74 MiB, resulting in a net saved memory for backward (residuals/activations) of 857.33 MiB. The top 5 memory-allocating operations during forward are `aten::mul`, `aten::sub`, `aten::div`, `aten::empty` and `aten::exp`. Each operation allocates exactly 128.0 MiB, each contributing 7.66% to the total block allocations. These operations directly correspond to intermediate attention matrix operations of shape `[4, 32, 512, 512]` in FP32, such as QK scaling and the numerically stable softmax computation.

    During the backward pass, saved activations are progressively freed while gradient tensors are emitted and retained on the parameters (`param.grad`). Across the entire model, net retained gradients total 13,028.45 MiB; after deducting 195.31 MiB for the embedding and final language modeling head, the estimated gradient memory per `TransformerBlock` is 401.04 MiB. This result fully matches theoretical expectations: in standard PyTorch training, each trainable parameter tensor produces a `.grad` tensor of the exact same shape and precision (FP32, 4 bytes/parameter). Thus, the 401.04 MiB gradient footprint precisely reflects the parameter memory size of an individual `xl` transformer block (approximately 100M parameters).

    #figure(
      caption: [Reconstructed timeline],
      image("assets/memory_timeline_reconstructed.png"),
    )
  ]

== Memory-Optimal Gradient Checkpointing

Consider a Transformer with $N$ identical blocks stacked sequentially. Without any checkpointing, all $N$ blocks' worth of residuals are kept alive simultaneously, giving $O(N)$ peak activation memory. We have a free hand to wrap any subset of the forward pass in checkpoint, including nesting checkpoint calls inside one another.

+ What checkpointing strategy minimizes peak activation memory, ignoring the compute cost? Describe how you would arrange the checkpoint calls (a code sketch is fine), and give the asymptotic peak activation memory and compute of your strategy as a function of $N$. Assume the residuals saved by a single block dominate any per-checkpoint bookkeeping.

  *Deliverable*: A 3-5 sentence description of the strategy and its asymptotic peak memory, plus a short code sketch.

  #response[
    To minimize peak activation memory ignoring compute cost, we apply recursive binary checkpointing ($b=2$). Distinguishing between the forward boundary tensor $A_"ckpt"$ and a single block's internal recomputation activation $A_"internal"$, the active memory along a depth-$L$ tree is $M(b) = (b - 1) / (ln b) ln(N) A_"ckpt" + A_"internal"$. Because recursive checkpointing can subdivide all the way down to a single leaf block, $A_"internal"$ enters purely as an additive constant independent of $b$; since $f(b) = (b - 1) / (ln b)$ is strictly increasing for $b >= 2$, binary splitting ($b=2$) strictly minimizes peak memory regardless of the $A_"ckpt" / A_"internal"$ ratio. This achieves an asymptotic peak memory of $cal(O)(log N)$ with $cal(O)(N log N)$ compute. In the extreme unconstrained-compute limit, recomputing each block directly from the initial input $x_0$ yields $cal(O)(1)$ memory at $cal(O)(N^2)$ compute.

    ```python
    def recursive_checkpoint(layers, x):
        if len(layers) == 1:
            return layers[0](x)
        mid = len(layers) // 2
        # Checkpoint midpoint; inner segment is recursively checkpointed during backward
        x_mid = torch.utils.checkpoint.checkpoint(
            lambda y: recursive_checkpoint(layers[:mid], y), x, use_reentrant=False
        )
        return recursive_checkpoint(layers[mid:], x_mid)
    ```
  ]

+ Consider the `xl` model config with batch size 4 and sequence length 2048 as above. If you only have the time/compute budget to run one step of recomputation (meaning you may not nest checkpoint calls), what is the best checkpointing strategy to reduce peak memory? Profile your run's peak memory to validate your hypothesis. Compare the peak memory of the next smaller and larger checkpointing block sizes to be sure.

  *Deliverable*: A 3-5 sentence description of your reasoning along with the measured peak memory for your strategy.

  #response[
    Under single-level activation checkpointing with group size $k$, peak activation memory follows $M(k) approx N / k A_"ckpt" + k A_"internal"$. Because intra-block activations ($A_"internal"$, dominated by attention matrices and MLP intermediate states) drastically outweigh boundary residual states ($A_"internal" >> A_"ckpt"$), the theoretical optimum consistently shifts to the finest granularity of per-layer checkpointing ($k^* = 1$). Our steady-state profiling confirms this across all settings: $k=1$ universally achieves the lowest peak memory ($51.77 "GiB"$ at $L=512$, $54.17 "GiB"$ at $L=1024$, and $63.44 "GiB"$ at the target $L=2048$). Furthermore, testing adjacent block sizes reveals that each increment in $k$ imposes an exact linear overhead ($approx +0.44$, $+1.77$, and $+5.94 "GiB"$ per step for $L=512, 1024,$ and $2048$, respectively), which scales quadratically with $L$ and proves that $k A_"internal"$ decisively dominates the memory trade-off. At $L=2048$, this rapid accumulation inflates memory from $63.44 "GiB"$ ($k=1$) to $69.37 "GiB"$ ($k=2$) and $75.31 "GiB"$ ($k=3$), with $k >= 4$ triggering OOM, establishing per-layer checkpointing ($k=1$) as the strictly optimal strategy.

    #figure(
      table(
        columns: (auto, auto, auto, auto, auto, auto, auto, auto),
        inset: (x: 7pt, y: 5.5pt),
        align: center + horizon,
        stroke: none,

        table.hline(stroke: 1.2pt),
        table.header(
          table.cell(rowspan: 2)[*Context*],
          table.cell(colspan: 7, align: center + bottom, inset: (
            bottom: 3pt,
          ))[*Checkpoint Group Size / Peak Memory (GiB)*],

          table.hline(start: 1, end: 8, stroke: 0.4pt),

          [*size = 1*], [*size = 2*], [*size = 3*], [*size = 4*], [*size = 5*], [*size = 6*], [*size = 7*],
        ),
        table.hline(stroke: 0.6pt),

        [512], [51.77], [52.20], [52.65], [53.09], [53.52], [53.97], [54.40],
        [1024], [54.17], [55.94], [57.72], [59.49], [61.27], [63.05], [64.82],
        [2048], [63.44], [69.37], [75.31], [OOM], [OOM], [OOM], [OOM],

        table.hline(stroke: 1.2pt),
      ),
      caption: [Peak memory usage (GiB) of the `xl` model under activation recomputation across varying checkpoint group sizes and context lengths (steady-state with 1 warmup step and 1 evaluated step).],
    ) <recompute_peak_memory>
  ]

= GPU Kernels

== PyTorch Attention Benchmarking

Benchmark your attention implementation at different scales. Write a script that will:

- Fix the batch size to 8 and don't use multihead attention (i.e. remove the head dimension).
- Iterate through the cartesian product of `[16, 32, 64, 128]` for the head embedding dimension $d_"model"$ , and `[256, 1024, 4096, 8192, 16384]` for the sequence length.
- Create random inputs $Q$, $K$, $V$ for the appropriate size.
- Time 100 forward passes through attention using the inputs.
- Measure how much memory is in use before the backward pass starts, and time 100 backward passes.
- Make sure to warm up, and to call `torch.cuda.synchronize()` after each forward/backward pass.

Depending on your GPU, some of these configurations are expected to run out of memory. Report the timings (or out-of-memory errors) you get for these configurations. At what size do you get out-of-memory errors? Do the accounting for the memory usage of attention in one of the smallest configurations you find that runs out of memory (you can use the equations for memory usage of Transformers from Assignment 1. How does the memory saved for backward change with the sequence length? What would you do to eliminate this memory cost?

*Deliverable*: A table with your timings, your calculations for the memory usage, and a 1-2 paragraph response.

#response[
  
]
