#import "@preview/ilm:2.1.1": *

#show: ilm.with(
  title: [CS336 Assignment 2],
  authors: "Morethan",
  date: datetime(year: 2026, month: 09, day: 04),
  abstract: [Language Modeling from Scratch - Systems and Parallelism],
  bibliography: bibliography("refs.bib"),
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
)

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
