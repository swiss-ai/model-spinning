:rocket: *New Tool Available: Model Spinning* :rocket:

Hey everyone! We've just released a new tool to make launching ML models on our SLURM cluster super easy:

*model-spinning* lets you quickly spin up models on Bristen or Clariden with a simple command:

```bash
spin-model.py --model <model-name> --account <your-account> --time 1h
```

:sparkles: *Features*:
• Launch models with vLLM or SP backends
• Automatically configures for Bristen/Clariden environments
• Simple time specification (1h, 90m, etc.)
• Custom environment variable support
• Easy access to model logs

:computer: *Installation*:
```bash
git clone https://github.com/swiss-ai/model-spinning.git
cd model-spinning
chmod +x spin-model.py
```

:zap: *Quick Install*:
```bash
wget https://raw.githubusercontent.com/swiss-ai/model-spinning/refs/heads/main/spin-model.py -O spin-model && chmod 755 spin-model && mv spin-model ~/.local/bin/
```

:speech_balloon: *Chat with your model*: https://fmapi.swissai.cscs.ch/chat

Try it out and let us know what you think! Check out the repo for more details and options. 