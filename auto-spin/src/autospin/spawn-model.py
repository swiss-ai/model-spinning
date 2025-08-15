import click
import yaml
from pydantic import BaseModel, ValidationError
from typing import Dict
import firecrest as f7t
from importlib import resources as imp_resources
from jinja2 import Environment, FileSystemLoader
from autospin import scripts
import os

AS_JOB_PREFIX:str="+as-"



class ModelConfig(BaseModel):
    instances: int = 0
    model_name: str
    model_path: str
    model_args: str
    sub_process:str
    environment:str
    serving_engine:str
    time_limit:str
    ocf_version:str


class Config(BaseModel):
    models: Dict[str, ModelConfig]
    client_id: str
    client_secret: str
    token_uri:str
    firecrest_uri:str
    system_name:str
    account:str
    bootstrap_addr:str


def load_config(file_path: str) -> Config:
    with open(file_path, 'r') as f:
        raw_data = yaml.safe_load(f)
    return Config.model_validate(raw_data)


def generate_jobs(config:Config):
    ids: Dict[str, str] = {}
    for model_id,model in config.models.items():
        for instance in range(model.instances):
            job_name=f"{AS_JOB_PREFIX}{model_id}-{instance}"
            parameters = model.model_dump()
            parameters["job_name"]=job_name
            parameters["bootstrap_addr"]=config.bootstrap_addr
            job_script = _build_script("spin.sh", parameters)
            ids[job_name]=job_script

    return ids

def _build_script(filename: str, parameters):

    script_environment = Environment(
        loader=FileSystemLoader(imp_resources.files(scripts)), autoescape=True
    )
    script_template = script_environment.get_template(filename)

    script_code = script_template.render(parameters)

    return script_code


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    """Loads and validates a YAML config file."""

    config = load_config(config_path)
    click.echo("Configuration loaded and validated successfully.")

    if config.client_secret.startswith("env:"):
        config.client_secret=os.getenv(config.client_secret.removeprefix("env:"))

    # Create an authorization object with Client Credentials authorization grant
    keycloak = f7t.ClientCredentialsAuth(
        config.client_id, config.client_secret, config.token_uri
    )

    # Setup the client for the specific account
    client: f7t.v2.Firecrest = f7t.v2.Firecrest(
        firecrest_url=config.firecrest_uri, authorization=keycloak
    )
    
    if config.system_name not in [item["name"] for item in client.systems()]:
            click.echo("❌ Unable to find the required cluster/system")
            return

    click.echo("Scanning for active jobs...")
    jobs = client.job_info(system_name=config.system_name, allusers=False)
    
    model_jobs = generate_jobs(config)

    active_jobs = [item for item in jobs if item["status"]["state"] in ["PENDING","RUNNING" ] ]
    autospin_active_jobs = [item for item in active_jobs if item["name"].startswith(AS_JOB_PREFIX)]
    
    running_models_name = [item["name"] for item in autospin_active_jobs if item["name"] in model_jobs.keys()]
    zombie_models  = [{"name":item["name"],"id":item["jobId"]} for item in autospin_active_jobs if item["name"] not in model_jobs.keys()]
    
    for job in running_models_name:
        click.echo(f"✅ job: {job} is running")

    missing_jobs_ids = [name for name in model_jobs.keys() if name not in running_models_name]

    for job in missing_jobs_ids:
        click.echo(f"❗ job: {job} is missing")

    for job in zombie_models:
        click.echo(f"💀 job: {job["name"]} is a zombie")
    
    if len(missing_jobs_ids) > 0:
        click.echo("Starting missing jobs...")
        for job_id in missing_jobs_ids: 
            client.submit(system_name=config.system_name, script_str=model_jobs[job_id],account=config.account,working_dir="/users/palmee/dispatcher")
            click.echo(f"🚀 job: {job_id} scheduled")
    
    if len(zombie_models) > 0:
        click.echo("Killing zombie jobs...")
        for job in zombie_models: 
            client.cancel_job(system_name=config.system_name, jobid=job["id"])
            click.echo(f"🔚 job: {job["name"]} canceled")


if __name__ == '__main__':
    main()
