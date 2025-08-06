# Auto Spin

Auto Spin is a GitHub workflow designed to monitor Slurm jobs running on a specified cluster and ensure that the required Large Language Model (LLM) models are continuously running.

## Github Workflow Setup

### LLM Models Jobs

To define the models that need to be served as Slurm jobs, you must configure them in the [config.yaml](./config.yaml) file. Submitting a commit with changes to the [config.yaml](./config.yaml) file will trigger the Auto Spin workflow, which will then synchronize the jobs on the HPC cluster.


### Workflow

The [Github workflow](https://github.com/swiss-ai/model-spinning/blob/main/.github/workflows/autospin.yml) is scheduled to run every 5 minutes (the interval may vary depending on the load on GitHub) or whenever the [config.yaml](./config.yaml) file is changed. You can monitor the workflow on the repository's [Actions page](https://github.com/swiss-ai/model-spinning/actions).


## Local Env Setup

It is also possible to run the Auto Spin CLI locally. To run Auto Spin on your local machine, you need a Firecrest client_id and a client_secret to authenticate yourself against the [CSCS Firecrest API](https://docs.cscs.ch/services/firecrest/). You can obtain the required client credentials on the [CSCS developer portal](https://docs.cscs.ch/services/firecrest/#cscs-developer-portal).


### Set-up Environment
```
cd auto-spin
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run autospin mannualy
```
export F7T_CLIENT_SECRET="[your firecrest api secret]"

cd auto-spin/src
source ../.venv/bin/activate

python -m autospin.spawn-model ../config.yaml
```