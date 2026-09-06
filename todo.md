Tasks:

- [ ] redo the exp1 with latest model size params
- [ ] update note file and commit

```sh
uv run cs336_systems/benchmarking_script.py --name basic_bench_w5_s50 --warm_up 5 --steps 50

uv run cs336_systems/benchmarking_script.py --name basic_bench_w0_s50 --warm_up 0 --steps 50
uv run cs336_systems/benchmarking_script.py --name basic_bench_w1_s50 --warm_up 1 --steps 50
uv run cs336_systems/benchmarking_script.py --name basic_bench_w2_s50 --warm_up 2 --steps 50

uv run cs336_systems/benchmarking_script.py --name basic_bench_w5_s150 --warm_up 5 --steps 150
uv run cs336_systems/benchmarking_script.py --name basic_bench_w5_s500 --warm_up 5 --steps 500
```

Expects:

- [ ] the logs in `benchmark_res` directory

---

Tasks:

- [ ] nsys basic exp, no monkey patch, check the correctness of code

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
  --gpu-metrics-devices=0 \
  --cuda-memory-usage=true \
  --stats=true \
  --force-overwrite=true \
  -- python cs336_systems/nsys_profile.py --warm_up 5 --steps 3

# no ui
OUT_DIR="profiles/$(date +'%Y%m%d_%H%M%S')" && mkdir -p "$OUT_DIR" && uv run nsys profile \
  -o "$OUT_DIR/profile" \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,cudnn,cublas,nvtx \
  --gpu-metrics-devices=0 \
  --cuda-memory-usage=true \
  --export=csv \
  --stats=true \
  --force-overwrite=true \
  -- python cs336_systems/nsys_profile.py --warm_up 5 --steps 3
```

Expects:

- [ ] log in `profile` directory and statistic printed in terminal

---


Tasks:

- [ ] longest context length experiment

```sh
uv run cs336_systems/benchmarking_script.py --name context_length --warm_up 5 --steps 5000 --context_length 1024
uv run cs336_systems/benchmarking_script.py --name context_length --warm_up 5 --steps 5000 --context_length 2048
uv run cs336_systems/benchmarking_script.py --name context_length --warm_up 5 --steps 5000 --context_length 4096
uv run cs336_systems/benchmarking_script.py --name context_length --warm_up 5 --steps 5000 --context_length 8192
```

Expects:

- [ ] longest power-of-two context length
