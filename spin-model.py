#!/usr/bin/env python3

import argparse
import os
import random
import re
import subprocess
import sys
import requests
# from bs4 import BeautifulSoup
# import redis


# redis_client = redis.Redis(
#     host='major-kid-42419.upstash.io',
#     port=6379,
#     password=keys.REDIS_PASSWORD,
#     ssl=True
# )


# def parse_hf_logo(hf_username):
#     url = f"https://huggingface.co/{hf_username}"
#     response = requests.get(url)
#     if response.status_code != 200:
#         print(f"Failed to load HF page for {hf_username}")
#         return None

#     soup = BeautifulSoup(response.text, 'html.parser')

#     # Find the first <img> tag with all the specified classes
#     img = soup.select_one("img.h-full.w-full.rounded-lg.object-cover")

#     if img and img.has_attr('src'):
#         return img['src']
#     else:
#         print("Avatar image not found.")
#         return None


# def save_model_logo(model):
#     org = model.split('/')[0]
#     img_url = parse_hf_logo("https://huggingface.co/" + org)
#     redis_client.set(org, img_url)


def parse_duration(time_str):
    """Convert various time formats to SLURM format (HH:MM:SS)."""
    # Check if already in SLURM format (HH:MM:SS)
    if re.match(r"^\d+:\d+:\d+$", time_str):
        return time_str

    hours, minutes, seconds = 0, 0, 0
    remaining = time_str

    # Extract time components
    while remaining:
        match = re.match(r"^(\d+)([hms]?)", remaining)
        if not match:
            break

        num = int(match.group(1))
        unit = match.group(2)

        if unit == "h" or unit == "":
            hours += num
        elif unit == "m":
            minutes += num
        elif unit == "s":
            seconds += num

        remaining = remaining[len(match.group(0)):]

    # Normalize time (carry over)
    minutes += seconds // 60
    seconds %= 60
    hours += minutes // 60
    minutes %= 60

    return f"{hours}:{minutes:02d}:{seconds:02d}"


def run_cmd(cmd, shell=False):
    """Run a subprocess command and print both stdout and stderr"""
    print(f"Running command: {cmd}")
    if isinstance(cmd, list) and shell:
        cmd = " ".join(cmd)
    proc = subprocess.run(
        cmd,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    print(f"STDOUT: {proc.stdout}")
    if proc.stderr:
        print(f"STDERR: {proc.stderr}")
    
    return proc


def get_help_content(filename):
    repo = "https://raw.githubusercontent.com/swiss-ai/model-spinning/refs/heads/main"
    url = f"{repo}/{filename}"
    
    try:
        # Try to fetch from GitHub first
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        
        print(f"Failed to fetch {url}, status: {response.status_code}")
    except Exception as e:
        print(f"Failed to fetch from GitHub: {e}")
    

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Launch a model on SLURM")
    parser.add_argument("--model", help="Name of the model to launch")
    parser.add_argument("--time", default="1h", help="Time duration for the job. Examples: 2h, 1h30m, 90m, 1:30:00")
    parser.add_argument("--vllm", action="store_true", help="Use vllm instead of sp to serve the model")
    parser.add_argument("--vllm-help", action="store_true", help="Show available options for **vllm** model server")
    parser.add_argument("--sp-help", action="store_true", help="Show available options for **sp** model server")
    parser.add_argument("--account", help="Slurm account to use for job submission")
    parser.add_argument("--env", action="append", help="Specify environment variables in format KEY=VALUE", default=[])
    parser.add_argument("--environment", help="Specify a custom environment file path")
    # Parse only known arguments, leaving the rest for the sp command
    args, extra_args = parser.parse_known_args()
    served_model_name = next((extra_args[i+1] for i, arg in enumerate(extra_args) if arg == '--served-model-name'), None)
    if served_model_name and '--' in served_model_name:
        print("[Warning] The served model name contains dashes (--). This may cause issues.")
    
    if args.vllm_help:
        print(get_help_content("vllm-docs.txt"))
        sys.exit(0)

    if args.sp_help:
        print(get_help_content("sp-docs.txt"))
        sys.exit(0)

    # Step 4: Require --model manually
    if not args.model:
        print("Error: --model is required.\n")
        parser.print_help()
        sys.exit(1)
    
    # Handle account
    if not args.account:
        print("No account specified. Checking available accounts...")
        cmd = "sacctmgr show associations user=$USER format=user,account%20"
        run_cmd(cmd, shell=True)
        print("Please specify an account with --account")
        sys.exit(1)

    # Store model name
    model = args.model

    print(f"Served model name: {served_model_name if served_model_name else model}")

    # if not served_model_name:
    #     save_model_logo(model)

    # Get hostname to identify node type
    hostname = run_cmd("hostname", shell=True).stdout.strip()

    # Set partition and show message based on hostname
    if hostname.startswith("nid"):
        node = "bristen"
        print("It's bristen so be aware that the time is limited and you can run a model up to 1 hour. While on clariden up to 24")
        # partition = "debug"
        # PARTITION = "#SBATCH --partition=debug"
        PARTITION = ""
        ocf_command = "/ocfbin/ocf-v2"
        NCCL_SO_PATH = "export SP_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/" 
        ENV_TOML = args.environment if args.environment else "/capstor/store/cscs/swissai/a09/xyao/llm_service/sp.toml"
    elif hostname.startswith("clariden"):
        node = "clariden"
        print("Clariden node is used")
        # partition = "normal"
        PARTITION = "#SBATCH --partition=normal"
        ocf_command = '/ocfbin/ocf-arm'
        NCCL_SO_PATH = "export SP_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/" 
        ENV_TOML = args.environment if args.environment else "/capstor/store/cscs/swissai/a09/xyao/llm_service/clariden/sp-arm.toml"
        if "apertus" in model.lower() or any("apertus" in arg.lower() for arg in extra_args):
            print("[Warning] It seems like you're trying to launch Apertus on Clariden. Please note that we can run Apertus only on Bristen.")
    else:
        print("It's neither clariden nor bristen node. Aborting")
        sys.exit(1)
    
    # Default values
    home_dir = os.environ.get('HOME')
    if not home_dir:
        print("Warning: HOME environment variable not set")
        random_bytes = random.randbytes(8).hex()
        home_dir = f"/tmp/{random_bytes}"
        try:
            os.makedirs(home_dir, exist_ok=True)
            print(f"Created temporary directory at: {home_dir}")
        except Exception as e:
            print(f"Failed to create temporary directory: {e}")
            sys.exit(1)
    logs_dir = os.path.join(home_dir, 'spinning-logs')
    
    # Create logs directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(os.path.join(logs_dir, 'slurm_scripts'), exist_ok=True)

    # Convert timeout to SLURM format
    slurm_time = parse_duration(args.time)
    print(f"Time allocated for model: {slurm_time}")


    # Construct the full command for serving
    if args.vllm:
        if node == "bristen": 
            print("VLLM is not supported on Bristen, try to run this script on Clariden")
            exit(1)

        # Rewrite an NCCL.so path for vllm on clariden
        NCCL_SO_PATH = "export VLLM_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/"
        if not args.environment:
            ENV_TOML = "/capstor/store/cscs/swissai/a09/xyao/llm_service/clariden/vllm.toml"


    serve_command = f"{'vllm' if args.vllm else 'sp'} serve {model} --host 0.0.0.0 --port 8080 {' '.join(extra_args)}"

    # Print the constructed command
    print("Command for serving:")
    print(serve_command)

    # Process environment variables from --env
    env_vars = ""
    for env_var in args.env:
        if "=" in env_var:
            env_vars += f"export {env_var}\n"
        else:
            print(f"Warning: Ignoring invalid environment variable format: {env_var}")

    # Create SLURM script content
    log_files = f"{logs_dir}/model-logs-%j"

    job_script = f"""#!/bin/bash
#SBATCH --job-name={'vllm' if args.vllm else 'sp'}-{served_model_name if served_model_name else model}
#SBATCH --output={log_files}.out
#SBATCH --error={log_files}.err
#SBATCH --container-writable
#SBATCH --time={slurm_time}
#SBATCH --ntasks-per-node=1
#SBATCH --dependency=singleton
#SBATCH --account={args.account}
#SBATCH --environment={ENV_TOML}
{PARTITION}

export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export PROMETHEUS_MULTIPROC_DIR=/ocfbin/scratch
{NCCL_SO_PATH}
{env_vars}

{ocf_command} start --bootstrap.addr /ip4/148.187.108.172/tcp/43905/p2p/QmP3tRgnWak28wH27soURViY8WVNB5rqYbciacRRMMaXu6 --subprocess "{serve_command}" \\
    --service.name llm \\
    --service.port 8080
"""

    # Generate a random script name
    rand_suffix = ''.join(random.choices('0123456789abcdef', k=10))
    script_path = f"{logs_dir}/slurm_scripts/run_{rand_suffix}.sh"
    
    # Save the script to a file
    with open(script_path, "w") as f:
        f.write(job_script)
    
    # Make the script executable
    os.chmod(script_path, 0o755)

    # Submit the job and capture the output
    sbatch_result = run_cmd(["sbatch", script_path])
    sbatch_output = sbatch_result.stdout.strip()
    print("sbatch command:", ["sbatch", script_path])
    print("sbatch output:", sbatch_output)

    # Extract the job ID from the output
    jobid_match = re.search(r'\d+', sbatch_output)
    if jobid_match:
        jobid = jobid_match.group(0)
    else:
        print("Warning: Could not extract job ID")
        jobid = "unknown"

    # Clean up the script after submission
    # os.remove(script_path)

    print(f"""
Job submitted. To know estimated time of start, run:
  squeue --me --start

Your job ID is: {jobid}

To get more information about the job: 
  scontrol show job {jobid}

To cancel this job/model, run:
  scancel {jobid}

To view logs for this job:
  cat {log_files.replace('%j', jobid)}.out  # For stdout
  cat {log_files.replace('%j', jobid)}.err  # For stderr

Chat with model when it's ready: https://fmapi.swissai.cscs.ch/chat
""")


if __name__ == "__main__":
    main() 
