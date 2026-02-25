#!/usr/bin/env bash
# Quick sanity check: verify model loads and can generate
# Run this first on the GPU server to catch issues early

set -euo pipefail

source .venv/bin/activate 2>/dev/null || true

MODEL="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"

echo "=== Sanity Check ==="
echo "Model: $MODEL"

python3 -c "
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')

print()
print('Loading processor...')
processor = AutoProcessor.from_pretrained('$MODEL', trust_remote_code=True)
print('Loading model...')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    '$MODEL',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)
print(f'Model loaded on: {model.device}')
print(f'Model dtype: {model.dtype}')

# Test generation
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'What is 2+2? Put your reasoning in <think> tags and answer in <answer> tags.'},
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors='pt').to(model.device)

print('Generating...')
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
response = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

print(f'Response: {response}')
print()
print('Sanity check PASSED')
"

echo ""
echo "=== Sanity check complete ==="
