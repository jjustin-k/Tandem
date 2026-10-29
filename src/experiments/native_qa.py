# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "huggingface-hub==0.36.0", "certifi==2025.10.5"]
# ///
"""Native QA control on the paired HotpotQA validation prompts.

uv run --script src/experiments/native_qa.py --device cuda --output artifacts/native-qa.json

Downloads the pinned models and the prompt manifest/labels. Output contains
answers and predictions; keep it out of Git. GPU use is restricted by setting
CUDA_VISIBLE_DEVICES before running this script on a shared server.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import string
from pathlib import Path

import certifi
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = 'tandemkv/paired-kv-qwen-smollm'
REVISION = 'df88b670d7d71b77eeffdd873143c40d62b8306b'
PREFIX = 'runs/hotpot-200-20261028'
MANIFEST_SHA256 = 'b88a13e01fd31f9fe56e2dd05b10e6703279080d3669f033c131ecfcdb0b9532'
LABELS_SHA256 = '99fbad2b671a5df2598e0e612fc9de709c9daac7492b246943b274096111e5eb'
ANSWER_CUE = ' '
MODELS = {
    'qwen': {'id': 'Qwen/Qwen2.5-0.5B-Instruct', 'revision': '7ae557604adf67be50417f59c2c2f167def9a775', 'type': 'qwen2'},
    'smol': {'id': 'HuggingFaceTB/SmolLM2-360M-Instruct', 'revision': 'a10cc1512eabd3dde888204e902eca88bddb4951', 'type': 'llama'},
}


def normalize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def score(prediction, answer):
    predicted, gold = normalize(prediction), normalize(answer)
    exact = float(predicted == gold)
    # HotpotQA's yes/no/noanswer labels must match exactly.
    if (predicted in ('yes', 'no', 'noanswer') or gold in ('yes', 'no', 'noanswer')) and not exact:
        return exact, 0.0
    p, g = predicted.split(), gold.split()
    overlap = sum((collections.Counter(p) & collections.Counter(g)).values())
    if not p or not g:
        return exact, float(p == g)
    if not overlap:
        return exact, 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return exact, 2 * precision * recall / (precision + recall)


def load_inputs(manifest_path, labels_path):
    manifest_bytes, labels_bytes = manifest_path.read_bytes(), labels_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256 or hashlib.sha256(labels_bytes).hexdigest() != LABELS_SHA256:
        raise ValueError('Input file checksum mismatch')
    manifest, labels = json.loads(manifest_bytes), json.loads(labels_bytes)
    if manifest['models'] != MODELS or labels['manifest_sha256'] != MANIFEST_SHA256:
        raise ValueError('Dataset model or manifest identity mismatch')
    samples = manifest['samples']
    if len(samples) != 200 or len({s['id'] for s in samples}) != 200:
        raise ValueError('Unexpected or duplicate sample IDs')
    answers = {row['id']: row for row in labels['labels']}
    if len(answers) != len(labels['labels']) or set(answers) != {s['id'] for s in samples}:
        raise ValueError('Labels do not match prompt IDs')
    for sample in samples:
        row = answers[sample['id']]
        if row['split'] != sample['split'] or row['source_row'] != sample['source_row'] or not row['answer'].strip():
            raise ValueError('Label provenance mismatch')
        if hashlib.sha256(sample['text'].encode()).hexdigest() != sample['text_sha256']:
            raise ValueError('Prompt text checksum mismatch')
    return samples, answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cuda')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--limit', type=int, help='Limit validation prompts for a smoke test')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new path')
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Token and sample limits must be positive')
    if len(set(args.models)) != len(args.models):
        parser.error('Do not repeat a model')
    if args.device == 'cuda' and (not os.environ.get('CUDA_VISIBLE_DEVICES') or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        parser.error('Set CUDA_VISIBLE_DEVICES to one allocated GPU')
    torch.set_num_threads(4)
    def fetch(name):
        return Path(hf_hub_download(REPO, f'{PREFIX}/{name}', repo_type='dataset', revision=REVISION))
    samples, answers = load_inputs(fetch('manifest.json'), fetch('labels.json'))
    selected = [sample for sample in samples if sample['split'] == 'validation']
    if len(selected) != 40:
        raise ValueError('Expected 40 validation prompts')
    if args.limit:
        selected = selected[:args.limit]
    result = {'dataset': REPO, 'dataset_revision': REVISION, 'manifest_sha256': MANIFEST_SHA256,
              'labels_sha256': LABELS_SHA256, 'split': 'validation', 'prompt_count': len(selected),
              'device': args.device, 'torch': torch.__version__, 'max_new_tokens': args.max_new_tokens,
              'decoding': 'greedy', 'answer_cue': ANSWER_CUE,
              'answer_extraction': 'first generated line', 'models': {}}
    for alias in args.models:
        spec = MODELS[alias]
        tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'])
        # Tokenize the cue separately: the saved prompt token IDs must remain an
        # exact prefix for comparison with replayed or translated caches.
        cue_ids = tokenizer(ANSWER_CUE, add_special_tokens=False)['input_ids']
        if not cue_ids:
            raise ValueError('Answer cue did not produce any tokens')
        model = AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'], dtype=torch.float32, attn_implementation='eager').eval().to(args.device)
        rows = []
        for sample in selected:
            token_ids = tokenizer(sample['text'], add_special_tokens=False)['input_ids']
            if token_ids != sample['tokens'][alias]['ids']:
                raise ValueError(f'Tokenizer mismatch: {sample["id"]}')
            tokens = torch.tensor([token_ids + cue_ids], device=args.device)
            with torch.inference_mode():
                generated = model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                    max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id)
            continuation = generated[0, len(token_ids) + len(cue_ids):]
            decoded = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            prediction = decoded.split('\n', 1)[0].strip()
            answer = answers[sample['id']]['answer']
            exact, f1 = score(prediction, answer)
            rows.append({'id': sample['id'], 'answer': answer, 'prediction': prediction, 'generated_text': decoded,
                         'generated_tokens': len(continuation), 'reached_token_limit': len(continuation) == args.max_new_tokens,
                         'exact_match': exact, 'f1': f1})
            print(f'{alias} {len(rows)}/{len(selected)}: {sample["id"]} EM={exact:.0f} F1={f1:.3f}', flush=True)
        result['models'][alias] = {'model': spec, 'cue_token_ids': cue_ids,
                                   'exact_match': sum(r['exact_match'] for r in rows) / len(rows),
                                   'f1': sum(r['f1'] for r in rows) / len(rows), 'samples': rows}
        del model, tokenizer
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'Saved baseline: {args.output}', flush=True)


if __name__ == '__main__':
    main()
