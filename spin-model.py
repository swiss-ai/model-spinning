#!/usr/bin/env python3
# Interactive usage: spin-model *

import argparse
import os
import random
import re
import subprocess
import sys
import requests

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

def print_warning(msg):
    """Print warning message in red"""
    print(f"{RED}{msg}{RESET}")

def print_success(msg):
    """Print success message in green"""
    print(f"{GREEN}{msg}{RESET}")


def get_saved_account():
    """Load saved account from config file"""
    config_file = os.path.expanduser("~/.spin-model-config")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                for line in f:
                    if line.startswith('SPIN_MODEL_ACCOUNT='):
                        return line.split('=', 1)[1].strip().strip('"\'')
        except Exception as e:
            print_warning(f"Warning: Could not read config file: {e}")
    return None


def save_account(account):
    """Save account to config file"""
    config_file = os.path.expanduser("~/.spin-model-config")
    try:
        # Read existing config
        existing_lines = []
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                existing_lines = [line for line in f if not line.startswith('SPIN_MODEL_ACCOUNT=')]
        
        # Write back with new account
        with open(config_file, 'w') as f:
            f.writelines(existing_lines)
            f.write(f'SPIN_MODEL_ACCOUNT="{account}"\n')
        
        print_success(f"Account '{account}' saved to {config_file}")
    except Exception as e:
        print_warning(f"Warning: Could not save account: {e}")


def get_user_accounts():
    """Get list of available SLURM accounts for current user"""
    try:
        cmd = "sacctmgr show associations user=$USER format=account%20 -n -P"
        result = run_cmd(cmd, shell=True)
        if result.returncode == 0:
            accounts = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            # Remove duplicates while preserving order
            seen = set()
            unique_accounts = []
            for account in accounts:
                if account not in seen:
                    seen.add(account)
                    unique_accounts.append(account)
            return unique_accounts
        else:
            print_warning("Could not retrieve accounts using sacctmgr")
            return []
    except Exception as e:
        print_warning(f"Error getting accounts: {e}")
        return []


def interactive_account_selection():
    """Interactive account selection interface"""
    print_success("Getting your available SLURM accounts...")
    accounts = get_user_accounts()
    
    if not accounts:
        print_warning("No accounts found. Please contact your system administrator.")
        return None
    
    print_success("Available accounts:")
    for i, account in enumerate(accounts, 1):
        print(f"  {i}. {account}")
    
    while True:
        try:
            choice = input(f"\nSelect account (1-{len(accounts)}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(accounts):
                    selected_account = accounts[idx]
                    save_account(selected_account)
                    return selected_account
                else:
                    print_warning(f"Please enter a number between 1 and {len(accounts)}")
            else:
                print_warning("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None
        except EOFError:
            print("\nExiting...")
            return None


# General default configuration
GENERAL_CONFIG = {
    "--host": "0.0.0.0",
    "--port": 8080,
    "--max-prefill-tokens": 8192,
    "--context-length": 8192,
    "--tp-size": 4
}

# Model registry with engine, tensor parallelism, and serving kwargs
MODEL_REGISTRY = {
    1: {
        "name": "llama",
        "path": "meta-llama/Llama-3.3-70B-Instruct",
        "engine": "python3 -m sglang.launch_server",
        "node": "bristen",
        "environment": "/capstor/store/cscs/swissai/a09/xyao/llm_service/sgl.toml",
        "kwargs": {
            "--model-path": "meta-llama/Llama-3.3-70B-Instruct",
            "--max-prefill-tokens": 32768,
            "--context-length": 32768,
            "--host": "localhost",
            "--reasoning-parser": "qwen3",
            "--tool-call-parser": "qwen25"
        }
    },
    2: {
        "name": "qwen",
        "path": "Qwen/Qwen3-32B",
        "engine": "python3 -m sglang.launch_server",
        "node": "bristen",
        "environment": "/capstor/store/cscs/swissai/a09/xyao/llm_service/sgl.toml",
        "kwargs": {
            "--model-path": "Qwen/Qwen3-32B",
            "--max-prefill-tokens": 32768,
            "--context-length": 32768,
            "--host": "localhost",
            "--reasoning-parser": "qwen3",
            "--tool-call-parser": "qwen25"
        }
    },
    3: {
        "name": "apertus-10T",
        "path": "/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus3-70B_iter_858000-tulu3-sft/checkpoint-13446",
        "engine": "sp serve",
        "node": "bristen",
        "environment": "/capstor/store/cscs/swissai/a09/xyao/llm_service/sp-dev.toml",
        "kwargs": {
            "--served-model-name": "swissai/apertus3-70b-10T-sft"
        }
    },
    4: {
        "name": "apertus-9T",
        "path": "/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus3-70B_iter_798250-tulu3-sft/checkpoint-13446",
        "engine": "sp serve",
        "node": "bristen",
        "environment": "/capstor/store/cscs/swissai/a09/xyao/llm_service/sp-dev.toml",
        "kwargs": {
            "--served-model-name": "swissai/apertus3-70b_iter_798250"
        }
    },
    5: {
        "name": "apertus-8B",
        "path": "/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus3-8B_iter_1678000-tulu3-sft",
        "engine": "sp serve",
        "node": "bristen",
        "environment": "/capstor/store/cscs/swissai/a09/xyao/llm_service/sp-dev.toml",
        "kwargs": {
            "--served-model-name": "swissai/apertus3-8b_iter_1678000"
        }
    },
    6: {
        "name": "apertus-70B-iter1155828",
        "path": "/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus3-70B_iter_1155828-tulu3-sft/checkpoint-13446",
        "engine": "sp serve",
        "node": "bristen",
        "environment": "/capstor/store/cscs/swissai/a09/xyao/llm_service/sp-dev.toml",
        "kwargs": {
            "--served-model-name": "swissai/apertus3-70b_iter_1155828"
        }
    }
}


def list_models():
    """Display available models in the registry"""
    print_success("Available models:")
    for model_id, config in MODEL_REGISTRY.items():
        engine_color = GREEN if config['engine'] == 'sgl' else '\033[94m' if config['engine'] == 'vllm' else '\033[93m'
        print(f"  {model_id}. {config['name']} ({config['path']})")
        # Get TP size from combined config or default to "N/A" if not specified
        model_kwargs = {**GENERAL_CONFIG, **config["kwargs"]}
        tp_size = model_kwargs.get("--tp-size", "N/A")
        print(f"     Engine: {engine_color}{config['engine']}{RESET}, TP: {tp_size}")
        kwargs_str = ", ".join([f"{k}={v}" for k, v in config['kwargs'].items()])
        print(f"     Args: {kwargs_str}")
        print()


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


def submit_job(job_script, logs_dir):
    """Saves the job script, submits it with sbatch, and returns the job ID."""
    rand_suffix = ''.join(random.choices('0123456789abcdef', k=10))
    script_path = os.path.join(logs_dir, 'slurm_scripts', f'run_{rand_suffix}.sh')
    
    with open(script_path, "w") as f:
        f.write(job_script)
    
    os.chmod(script_path, 0o755)

    sbatch_result = run_cmd(["sbatch", script_path])
    sbatch_output = sbatch_result.stdout.strip()
    
    jobid_match = re.search(r'\d+', sbatch_output)
    if jobid_match:
        return jobid_match.group(0)
    else:
        print_warning(f"Warning: Could not extract job ID from sbatch output: {sbatch_output}")
        return None


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Launch a model on SLURM")
    parser.add_argument("--model", help="Name of the model to launch (deprecated, use -m)")
    parser.add_argument("-m", "--model-id", type=int, help="Model ID from registry (use -l to list available models)")
    parser.add_argument("-l", "--list", action="store_true", help="List available models in registry")
    parser.add_argument("--login", action="store_true", help="Interactive account selection and save for future use")
    parser.add_argument("-t", "--time", default="1h", help="Time duration for the job. Examples: 2h, 1h30m, 90m, 1:30:00")
    parser.add_argument("-n", "--num-instances", type=int, default=1, help="Number of model instances to launch.")
    parser.add_argument("--vllm", action="store_true", help="Use vllm instead of sp to serve the model (overrides registry)")
    parser.add_argument("--sgl", action="store_true", help="Use sglang instead of sp to serve the model (overrides registry)")
    parser.add_argument("--vllm-help", action="store_true", help="Show available options for **vllm** model server")
    parser.add_argument("--sp-help", action="store_true", help="Show available options for **sp** model server")
    parser.add_argument("-a", "--account", help="Slurm account to use for job submission")
    parser.add_argument("-v", "--var", action="append", help="Specify environment variables in format KEY=VALUE", default=[])
    parser.add_argument("-e", "--environment", help="Specify a custom environment file path")
    # Parse only known arguments, leaving the rest for the sp command
    args, extra_args = parser.parse_known_args()
    served_model_name = next((extra_args[i+1] for i, arg in enumerate(extra_args) if arg == '--served-model-name'), None)
    if served_model_name and '--' in served_model_name:
        print_warning("[Warning] The served model name contains dashes (--). This may cause issues.")
    
    if args.vllm_help:
        print(get_help_content("vllm-docs.txt"))
        sys.exit(0)

    if args.sp_help:
        print(get_help_content("sp-docs.txt"))
        sys.exit(0)

    if args.list:
        list_models()
        sys.exit(0)
    
    if args.login:
        account = interactive_account_selection()
        if account:
            print_success(f"Account '{account}' has been saved and will be used by default in future runs.")
        sys.exit(0)
    
    # Check for interactive mode (spin-model *)
    if len(sys.argv) == 2 and sys.argv[1] == '*':
        print_success("Interactive mode activated!")
        account = interactive_account_selection()
        if not account:
            sys.exit(1)
        
        # Show model list and get selection
        list_models()
        while True:
            try:
                model_choice = input(f"\nSelect model (1-{len(MODEL_REGISTRY)}): ").strip()
                if model_choice.isdigit():
                    model_id = int(model_choice)
                    if model_id in MODEL_REGISTRY:
                        # Set args as if they were passed via command line
                        args.model_id = model_id
                        args.account = account
                        break
                    else:
                        print_warning(f"Please enter a number between 1 and {len(MODEL_REGISTRY)}")
                else:
                    print_warning("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
            except EOFError:
                print("\nExiting...")
                sys.exit(0)

    # Require either --model or -m
    if not args.model and not args.model_id:
        print_warning("Error: Either --model or -m (model ID) is required. Use -l to list available models.\n")
        parser.print_help()
        sys.exit(1)
    
    if args.model_id and args.model_id not in MODEL_REGISTRY:
        print_warning(f"Error: Model ID {args.model_id} not found in registry. Use -l to list available models.")
        sys.exit(1)
    
    # Handle account
    if not args.account:
        # Try to load saved account
        saved_account = get_saved_account()
        if saved_account:
            args.account = saved_account
            print_success(f"Using saved account: {saved_account}")
        else:
            print_warning("No account specified. Use --login to set up an account, or specify with -a/--account")
            accounts = get_user_accounts()
            if accounts:
                print("Available accounts:")
                for account in accounts:
                    print(f"  - {account}")
            print_warning("Please specify an account with -a/--account or use --login for interactive setup")
            sys.exit(1)

    # Store model information
    if args.model_id:
        model_config = MODEL_REGISTRY[args.model_id]
        model = model_config["path"]
        model_name = model_config["name"]
        default_engine = model_config["engine"]
        
        # Union: general | model
        model_kwargs = GENERAL_CONFIG | model_config["kwargs"]
        tp_size = model_kwargs.get("--tp-size", 4)
        
        # Handle engine-specific argument mapping
        extra_args_str = ' '.join(extra_args)
        engine_name = default_engine.split()[0] if default_engine else "sp"
        
        if engine_name == "vllm" and "--tp-size" in extra_args_str:
            extra_args_str = extra_args_str.replace("--tp-size", "--tensor-parallel-size")
        
        extra_args = extra_args_str.split() if extra_args_str else []
    else:
        # Legacy mode using --model
        model = args.model
        model_name = args.model
        default_engine = "sp"
        tp_size = 1
        model_kwargs = {}

    print(f"Served model name: {served_model_name if served_model_name else (model_name if args.model_id else model)}")

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
        if args.model_id and "environment" in model_config:
            ENV_TOML = args.environment if args.environment else model_config["environment"]
        else:
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
            print_warning("[Warning] It seems like you're trying to launch Apertus on Clariden. Please note that we can run Apertus only on Bristen.")
    else:
        print("It's neither clariden nor bristen node. Aborting")
        sys.exit(1)
    
    # Default values
    home_dir = os.environ.get('HOME')
    if not home_dir:
        print_warning("Warning: HOME environment variable not set")
        random_bytes = random.randbytes(8).hex()
        home_dir = f"/tmp/{random_bytes}"
        try:
            os.makedirs(home_dir, exist_ok=True)
            print_success(f"Created temporary directory at: {home_dir}")
        except Exception as e:
            print_warning(f"Failed to create temporary directory: {e}")
            sys.exit(1)
    logs_dir = os.path.join(home_dir, 'spinning-logs')
    
    # Create logs directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(os.path.join(logs_dir, 'slurm_scripts'), exist_ok=True)

    # Convert timeout to SLURM format
    slurm_time = parse_duration(args.time)
    print_success(f"Time allocated for model: {slurm_time}")


    # Build serve command using engine from registry or override
    if args.model_id:
        engine_cmd = model_config["engine"]
        serve_args = [f"{key} {value}" for key, value in model_kwargs.items() if value is not True]
        
        if engine_cmd.startswith("sp serve"):
            serve_command = f"{engine_cmd} {model} {' '.join(serve_args)} {' '.join(extra_args)}"
        else:  # SGL
            serve_command = f"{engine_cmd} {' '.join(serve_args)} {' '.join(extra_args)}"
    else:
        # Legacy mode - determine engine from command line args
        if args.sgl:
            serve_command = f"python3 -m sglang.launch_server --model-path {model} --host localhost --port 8080 --tp-size 4 {' '.join(extra_args)}"
        elif args.vllm:
            serve_command = f"vllm serve {model} --host 0.0.0.0 --port 8080 {' '.join(extra_args)}"
        else:
            serve_command = f"sp serve {model} --host 0.0.0.0 --port 8080 {' '.join(extra_args)}"

    # Print the constructed command
    print_success("Command for serving:")
    print(serve_command)

    # Process environment variables from --var
    env_vars = ""
    for env_var in args.var:
        if "=" in env_var:
            env_vars += f"export {env_var}\n"
        else:
            print_warning(f"Warning: Ignoring invalid environment variable format: {env_var}")
    
    # Add SGL-specific environment variables if using SGL
    if args.model_id and "sglang" in model_config["engine"]:
        env_vars += "export PROMETHEUS_MULTIPROC_DIR=/ocfbin/scratch\n"
    elif not args.model_id and args.sgl:
        env_vars += "export PROMETHEUS_MULTIPROC_DIR=/ocfbin/scratch\n"

    # Create SLURM script content
    log_files = f"{logs_dir}/model-logs-%j"

    try:
        bootstrap_addr = requests.get("http://148.187.108.172:8092/v1/dnt/bootstraps").json()['bootstraps'][0]
    except (requests.exceptions.RequestException, KeyError, IndexError) as e:
        print_warning(f"Failed to fetch or parse bootstrap address: {e}. Using fallback.")
        bootstrap_addr = "/ip4/148.187.108.172/tcp/43905/p2p/Qma3y5cF2g39h9sJTTESy5AoatmsWUb9dEmaScvPUBg1fw"

    job_script_template = f"""#!/bin/bash
#SBATCH --job-name={'sgl' if args.sgl else 'vllm' if args.vllm else 'sp'}-{served_model_name if served_model_name else (model_name if args.model_id else model)}{"-%s" if args.num_instances > 1 else ""}
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
{NCCL_SO_PATH}
{env_vars}

{ocf_command} start --bootstrap.addr {bootstrap_addr} --subprocess "{serve_command}" \\
    --service.name llm \\
    --service.port 8080
"""

    jobids = []
    for i in range(1, args.num_instances + 1):
        jobid = submit_job(job_script_template.replace('%s', str(i)), logs_dir)
        jobids.append(jobid)

    print_success(f"""
Job submitted. To know estimated time of start, run:
  squeue --me --start

Your job ID is: {' '.join(map(str, jobids))}

To get more information about the job: 
  scontrol show job <jobid>

To cancel this job/model, run:
  scancel {' '.join(map(str, jobids))}

To view logs for this job:
  cat {log_files.replace('%j', "<jobid>")}.out  # For stdout
  cat {log_files.replace('%j', "<jobid>")}.err  # For stderr

Chat with model when it's ready: https://serving.swissai.cscs.ch/
""")


if __name__ == "__main__":
    main() 
