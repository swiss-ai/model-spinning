
## Set-up Environment
```
cd auto-spin
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run autospin mannualy
```
export F7T_CLIENT_SECRET="[your firecrest api secret]"

cd auto-spin/src
source ../.venv/bin/activate

python -m autospin.spawn-model ../config.yaml
```