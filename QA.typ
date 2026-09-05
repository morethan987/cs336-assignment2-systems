#import "@preview/ilm:2.1.1": *

#set text(lang: "en")

#show: ilm.with(
  title: [CS336 Assignment 2],
  authors: "Morethan",
  date: datetime(year: 2026, month: 09, day: 04),
  abstract: [Language Modeling from Scratch - Systems and Parallelism],
  bibliography: bibliography("refs.bib"),
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true),
)
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
    I measured the timing with 50 steps and 5 warm-up steps. A forward pass takes 20.65 ms with 2.66 ms standard deviation (relatively 12.88%), a backward pass takes 23.67 ms with 3.35 ms standard deviation (relatively 14.14%).
    #figure(
      caption: "Basic benchmarking results",
      table(
        columns: (auto, auto, auto),
        inset: (x: 8pt, y: 4.5pt),
        align: (left, center, center),
        stroke: none,

        table.hline(stroke: 1.2pt),
        [Stage], [Mean (ms)], [Std (ms)],
        table.hline(stroke: 0.6pt),

        [Prepare], [0.0902], [0.0259],
        [Forward], [20.6506], [2.6613],
        [Backward], [23.6743], [3.3483],
        [Optimizer], [2.8455], [0.3755],
        table.hline(stroke: 0.4pt),
        [Total], [47.2606], [6.1087],
        table.hline(stroke: 1.2pt),
      ),
    )
  ]

+ One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?

  *Deliverable:* A 2-3 sentence response.

  #response[
    Without warm-up steps, the mean latency at each stage increases markedly, and the total standard deviation is over 16 times higher than that with warm-up. This discrepancy is primarily attributed to the cold-start overheads of the GPU runtime and deep learning framework—such as CUDA context initialization, PyTorch memory caching allocator initialization (avoiding repeated `cudaMalloc`). Incorporating just 1 or 2 warm-up steps effectively eliminates these initialization artifacts, resulting in stable performance with significantly reduced standard deviation.

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

        [Prepare], [0.1695], [0.5768], [0.0843], [0.0169], [0.0845], [0.0176],
        [Forward], [38.2242], [127.3257], [19.8813], [4.0689], [19.9290], [3.6766],
        [Backward], [25.3251], [18.9235], [22.5781], [4.5535], [22.8588], [4.6035],
        [Optimizer], [3.0423], [2.1910], [2.7499], [0.5143], [2.8101], [0.4890],

        table.hline(stroke: 0.4pt),
        [Total], [66.7611], [148.5845], [45.2936], [8.9231], [45.6824], [8.5872],

        // bottom
        table.hline(stroke: 1.2pt),
      ),
      caption: [Warm-up ablation],
    )
  ]

== Nsight Systems Profiling

Profile your forward pass, backward pass, and optimizer step using nsys with two model sizes from Table 1 of your choice as well as three power-of-two context lengths larger than 128, where the largest available size should be the longest context length you can fit in memory. Pick the combinations you think would be the most interesting to look at. For each profile answer the following questions:

+ What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?

  *Deliverable*: A 1-2 sentence response.

  #response[]

+ What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes? (Hint: look at the “CUDA GPU Kernel Summary” under “Stats System View”, and filter using NVTX ranges to identify which parts of the model are responsible for which kernels.)

  *Deliverable*: A 1-2 sentence response.

  #response[]

+ Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?

  *Deliverable*: A 1-2 sentence response.

  #response[]

+ Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you’d do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?

  *Deliverable*: A 1-2 sentence response.

  #response[]

+ Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

  *Deliverable*: A 1-2 sentence response.

  #response[]
