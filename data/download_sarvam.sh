#!/bin/bash
# Download Sarvam-1 model from HuggingFace
huggingface-cli download sarvamai/sarvam-1 \
  --local-dir ../pretrained/sarvam-1

# Ensure add_eos_token is set (required for EOS-based embedding extraction)
# Sarvam-1 defaults to false; our pipeline requires true.
if command -v python3 &> /dev/null; then
  python3 -c "
import json, pathlib
p = pathlib.Path('../pretrained/sarvam-1/tokenizer_config.json')
cfg = json.loads(p.read_text())
if not cfg.get('add_eos_token'):
    cfg['add_eos_token'] = True
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print('Set add_eos_token=true in tokenizer_config.json')
else:
    print('add_eos_token already set to true')
"
fi
