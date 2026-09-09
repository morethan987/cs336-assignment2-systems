Tasks:

- [ ] Run toy model

```sh
uv run cs336_systems/toy.py
```

---

Tasks:

- [ ] time cost comparation.

```sh
uv run cs336_systems/benchmarking_script.py \
  --name mixed_precision_small \
  --warm_up 5 \
  --steps 50 \
  --model_size small \
  --use_mixed_precision

uv run cs336_systems/benchmarking_script.py \
  --name full_precision_small \
  --warm_up 5 \
  --steps 50 \
  --model_size small
```

---

Tasks:

- [ ] Remove backward and optimizer step from nsys benchmarking to check whether stand alone froward differs from complete traning loop.

```sh
OUT_DIR="profiles/medium_ctx512_forward_only/$(date +'%Y%m%d_%H%M%S')" && mkdir -p "$OUT_DIR" && uv run nsys profile \
  -o "$OUT_DIR/profile" \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,cudnn,cublas,nvtx \
  --cuda-memory-usage=true \
  --stats=true \
  --force-overwrite=true \
  -- python cs336_systems/nsys_profile.py --warm_up 5 --steps 5 --context_length 512
```

Expects:

- [ ] With less nsys injection, the forward pass may become a little bit faster but no notable difference.

---

Tasks:

- [ ] Warp the whole traning step only.

```sh
OUT_DIR="profiles/medium_ctx512_train_step/$(date +'%Y%m%d_%H%M%S')" && mkdir -p "$OUT_DIR" && uv run nsys profile \
  -o "$OUT_DIR/profile" \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,cudnn,cublas,nvtx \
  --cuda-memory-usage=true \
  --stats=true \
  --force-overwrite=true \
  -- python cs336_systems/nsys_profile.py --warm_up 5 --steps 5 --context_length 512
```

Expects:

- [ ] Get a comverged data instead of add data from different nvtx range.

---

Tasks:

- [ ] Attention insight.

```sh
OUT_DIR="profiles/medium_ctx512_att_insight/$(date +'%Y%m%d_%H%M%S')" && mkdir -p "$OUT_DIR" && uv run nsys profile \
  -o "$OUT_DIR/profile" \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,cudnn,cublas,nvtx \
  --cuda-memory-usage=true \
  --stats=true \
  --force-overwrite=true \
  -- python cs336_systems/nsys_profile.py --warm_up 5 --steps 5 --context_length 512
```

Expects:

- [ ] Get a resonable time costs attribution.

---

Tasks:

- [x] redo the exp1 with latest model size params
- [x] update note file and commit

```sh
uv run cs336_systems/benchmarking_script.py --name basic_bench_w5_s50 --warm_up 5 --steps 50

uv run cs336_systems/benchmarking_script.py --name basic_bench_w0_s50 --warm_up 0 --steps 50
uv run cs336_systems/benchmarking_script.py --name basic_bench_w1_s50 --warm_up 1 --steps 50
uv run cs336_systems/benchmarking_script.py --name basic_bench_w2_s50 --warm_up 2 --steps 50

uv run cs336_systems/benchmarking_script.py --name basic_bench_w5_s150 --warm_up 5 --steps 150
uv run cs336_systems/benchmarking_script.py --name basic_bench_w5_s500 --warm_up 5 --steps 500
```

Expects:

- [x] the logs in `benchmark_res` directory

---

Tasks:

- [x] nsys basic exp, no monkey patch, check the correctness of code

```sh
# for ui
OUT_DIR="profiles/$(date +'%Y%m%d_%H%M%S')" && mkdir -p "$OUT_DIR" && uv run nsys profile \
  -o "$OUT_DIR/profile" \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,cudnn,cublas,osrt,nvtx \
  --pytorch=functions-trace,autograd-shapes-nvtx \
  --cudabacktrace=all \
  --python-backtrace=cuda \
  --cuda-memory-usage=true \
  --stats=true \
  --force-overwrite=true \
  -- python cs336_systems/nsys_profile.py --warm_up 5 --steps 5

# no ui
OUT_DIR="profiles/$(date +'%Y%m%d_%H%M%S')" && mkdir -p "$OUT_DIR" && uv run nsys profile \
  -o "$OUT_DIR/profile" \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,cudnn,cublas,nvtx \
  --cuda-memory-usage=true \
  --stats=true \
  --force-overwrite=true \
  -- python cs336_systems/nsys_profile.py --warm_up 5 --steps 5
```

Expects:

- [x] log in `profile` directory and statistic printed in terminal

Comment:

- No need to run the first cmd, useless and takes much larger size.

---


Tasks:

- [x] longest context length experiment

```sh
uv run cs336_systems/benchmarking_script.py --name context_length --warm_up 5 --steps 5000 --context_length 1024
uv run cs336_systems/benchmarking_script.py --name context_length --warm_up 5 --steps 5000 --context_length 2048
uv run cs336_systems/benchmarking_script.py --name context_length --warm_up 5 --steps 5000 --context_length 4096
uv run cs336_systems/benchmarking_script.py --name context_length --warm_up 5 --steps 5000 --context_length 8192
```

Expects:

- [x] longest power-of-two context length: 2048

