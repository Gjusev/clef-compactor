"""Kaggle kernel: measure clef-compactor against the open-weights Clef model.

Follows the verified laya-evals kernel pattern (github.com/Gjusev/laya-evals):
clone the public repo, pip install it, run the measurement, print the verdict.
Outputs land in /kaggle/working/ and come back with `kaggle kernels output`.

Hardware: Kaggle T4 x2 (2 x 16 GB). We run Cloudflare/clef-flash (9B) in
float16 sharded across both GPUs with device_map="auto"; the 27B model does
not fit on this hardware. Credentials are never needed and never embedded:
the local backend loads weights from Hugging Face.
"""

import subprocess
import sys


def run(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


# 0. GPU sanity: Kaggle can silently boot without the GPU you asked for.
print("=== GPU check ===", flush=True)
run("nvidia-smi")

# 1. Repo + dependencies. transformers 5.10.2 is the version the Clef release
#    was tested with; Qwen3.5 support needs the 5.x line.
run("git clone --depth 1 https://github.com/Gjusev/clef-compactor.git")
run(f"{sys.executable} -m pip install -q --no-input ./clef-compactor")
run(f"{sys.executable} -m pip install -q --no-input 'transformers==5.10.2' accelerate safetensors")

run(f"{sys.executable} -c 'import torch, transformers; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"transformers\", transformers.__version__)'")

# 2. The measurement: full pipeline (batching, ranking, budget) against the
#    locally loaded clef-flash weights, scored on the committed gold dataset.
run(
    f"{sys.executable} clef-compactor/evals/run_eval.py "
    "--mode local --model-path Cloudflare/clef-flash --dtype float16 "
    "--out /kaggle/working/results-local.json --compare-laya"
)

# 3. Optional stretch: the 27B model in float16 needs ~54 GB and does not fit
#    on 2 x T4; skipped on purpose. If Kaggle ever offers bigger GPUs, run:
#    run_eval.py --mode local --model-path Cloudflare/clef

print("KERNEL COMPLETE", flush=True)
