# Distributed Retrieval-Augmented Generation

## Prerequisites

- Ubuntu 22.04
- Python 3.10
- Ollama 0.5.7

Install Python requirements:

```bash
$ pip install -r requirements.txt 
```

Install and start [Ollama](https://ollama.com/download/linux) local LLM service:

```bash
# start ollama service:
$ sudo systemctl start ollama
# start serving model "Llama 3.2 - 3B"
# for other Ollama models: https://ollama.com/library
$ ollama run llama3.2:3b
# preload a model into Ollama to get faster response times:
$ curl http://localhost:11434/api/generate -d '{"model": "llama3.2:3b"}'
# check ollama log
$ journalctl -u ollama
```

If you do not have root permission, install and start Ollama [manually](https://github.com/ollama/ollama/blob/main/docs/linux.md):

```bash
# Download and extract the package:
$ curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
$ sudo tar -C <your-path>/ -xzf ollama-linux-amd64.tgz
# Create user-mode systemd.service file `~/.config/systemd/user/ollama.service` with the following content:
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=<your-path>/bin/ollama serve
Restart=always
RestartSec=3
Environment="PATH=$PATH"

[Install]
WantedBy=default.target
# Start systemd ollama service in user mode:
$ systemctl --user enable ollama
$ systemctl --user start ollama
# Check ollama service log:
$ journalctl --user -f -u ollama
```

Install [NLTK Data](https://www.nltk.org/data.html)

```bash
$ python -m nltk.downloader all
```

## Models

The following Ollama models are utilized in our experiments.

- [Llama 3.2 3B](https://ollama.com/library/llama3.2:3b)
- [Gemma 2 2B](https://ollama.com/library/gemma2:2b)
- [Qwen 2.5 3B](https://ollama.com/library/qwen2.5:3b)

## Datasets

The following Hugging Face datasets are utilized in our experiments.

- [MMLU](https://huggingface.co/datasets/cais/mmlu)
- [medical_extended](https://huggingface.co/datasets/sarus-tech/medical_extended)
- [news-category-dataset](https://huggingface.co/datasets/heegyu/news-category-dataset)

> Use `HF_HUB_CACHE` to configure where repositories from the Hub will be cached locally (models, datasets and spaces).

## Run

```bash
$ python simulator.py
```

## DEBUG
