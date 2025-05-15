# Model Spinning

A utility for launching and serving AI models on SLURM clusters at CSCS.

## Quick Start

1. Download the script:
   ```bash
   wget https://raw.githubusercontent.com/swiss-ai/model-spinning/refs/heads/main/spin-model.py
   chmod 755 spin-model.py
   ```

2. Check your available SLURM accounts:
   ```bash
   sacctmgr show associations user=$USER format=user,account%20
   ```

3. Launch a model:


   ```bash
   # Launch Mistral 7B with tensor parallelism 2 for 30 minutes
   ./spin-model.py --model mistralai/Mistral-7B-Instruct-v0.3 --tp-size 2 --time 30m --account YOUR_ACCOUNT
   ```
 
## Usage

```
usage: spin-model.py [-h] [--model MODEL] [--time TIME] [--vllm] [--vllm-help]
                     [--sp-help] [--account ACCOUNT]

Launch a model on SLURM

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      Name of the model to launch
  --time TIME        Time duration for the job. Examples: 2h, 1h30m, 90m, 1:30:00
  --vllm             Use vllm instead of sp to serve the model
  --vllm-help        Show available options for **vllm** model server
  --sp-help          Show available options for **sp** model server
  --account ACCOUNT  Slurm account to use for job submission
```

Additional model-specific arguments can be passed after the main arguments.

## Important Parameters

### Model Serving Options

- **Model Server**: By default, the script uses the `sp` model server. For certain architectures, you can use `--vllm` to switch to the vLLM server.
- **Documentation**: 
  - [SP (SentencePiece) documentation](https://github.com/swiss-ai/mmore/blob/master/sp-docs.txt)
  - [vLLM documentation](https://github.com/swiss-ai/mmore/blob/master/vllm-docs.txt)
  - View these docs directly with:
    ```bash
    ./spin-model.py --sp-help    # For sp server options
    ./spin-model.py --vllm-help  # For vllm server options
    ```

### Tensor Parallelism

The `--tp-size` parameter specifies the tensor parallelism size when a model is too large to fit on a single GPU:

- Models < 2B parameters: `--tp-size 1`
- Models < 14B parameters: `--tp-size 2`
- Models < 45B parameters: `--tp-size 3`
- Models < 90B parameters: `--tp-size 4`

### Time Allocation

The `--time` parameter accepts various formats:
- `2h` (2 hours)
- `1h30m` (1 hour and 30 minutes)
- `90m` (90 minutes)
- `1:30:00` (1 hour and 30 minutes in SLURM format)

Note: On Bristen nodes, time is limited to 1 hour maximum, while Clariden nodes allow up to 24 hours.

## After Submission

Once your job is submitted, you'll see:
- Job ID
- Links to monitor your model
- Commands to check job status and logs 