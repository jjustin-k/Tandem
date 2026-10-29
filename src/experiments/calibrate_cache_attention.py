# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2", "huggingface-hub==0.36.0", "pyarrow==21.0.0", "certifi==2025.10.5"]
# ///
"""Calibrate the existing mapper on native receiver continuations, on one GPU."""
import argparse
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM
import paired_kv_dataset as base
from cache_attention import CacheAttention
from train_cache_attention import (Dataset, REPO, causal_positions, checkpoint,
                                   logit_kl, receiver_logits, translate_prefix, upload_outputs)
from evaluate_cache_attention import (CHECKPOINT, CHECKPOINT_REVISION, cached_start,
                                      check_destination, continue_greedy, lock_workdir,
                                      native_start, prefix_agreement)


def continuation_inputs(prompt, continuation):
    """Predict continuation[0] from the final prompt token, then each next token."""
    if not continuation:
        raise ValueError('Empty continuation')
    previous = torch.tensor([continuation[:-1]], dtype=prompt.dtype, device=prompt.device)
    return torch.cat([prompt, previous], -1), torch.cat([prompt[:, -1:], previous], -1)


def teacher_logits(receiver, prompt, continuation):
    full, _ = continuation_inputs(prompt, continuation)
    with torch.no_grad():
        return receiver(input_ids=full, use_cache=False,
                        logits_to_keep=len(continuation)).logits


def score(mapper, receiver, data, samples, traces, args, greedy=False):
    mapper.eval()
    rows = []
    with torch.no_grad():
        for sample in samples:
            ids = torch.tensor([sample['tokens'][args.target]['ids']], device=args.device)
            source = data.features(sample, args.source)
            target = data.features(sample, args.target)
            teacher = teacher_logits(receiver, ids, traces[sample['id']])
            _, suffix = continuation_inputs(ids, traces[sample['id']])
            control = receiver_logits(receiver, target[:-1].to(args.device), suffix)
            if not torch.allclose(control, teacher, rtol=1e-3, atol=1e-3):
                raise ValueError('Native continuation replay control failed')
            mapped = translate_prefix(mapper, source, sample, args, ids.shape[1] - 1)
            student = receiver_logits(receiver, mapped, suffix)
            row = {'id': sample['id'], 'kl': logit_kl(student, teacher).item(),
                   'teacher_path_top1': (student.argmax(-1) == teacher.argmax(-1)).float().mean().item(),
                   'control_max_logit_error': (control - teacher).abs().max().item()}
            if not torch.isfinite(torch.tensor(row['kl'])):
                raise ValueError('Nonfinite continuation evaluation loss')
            if greedy:
                output = cached_start(receiver, mapped, ids[:, -1:])
                generated = continue_greedy(receiver, output, args.tokens, args.eos_ids)
                row.update(prefix_agreement(traces[sample['id']], generated))
                row['native_tokens'], row['mapped_tokens'] = traces[sample['id']], generated
            rows.append(row)
    summary = {key: sum(r[key] for r in rows) / len(rows) for key in ('kl', 'teacher_path_top1')}
    if greedy:
        summary.update({key: sum(r[key] for r in rows) / len(rows)
                        for key in ('first_token_match', 'matching_prefix_tokens', 'sequence_equal')})
    return {'summary': summary, 'samples': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--pilot', action='store_true', help='Eight train / four validation prompts')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--tokens', type=int, default=32, help='Use 1 for the matched one-token calibration control')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=20261029)
    p.add_argument('--hours', type=float, default=2)
    p.add_argument('--upload-prefix', required=True)
    args = p.parse_args()
    args.device = 'cuda'
    if min(args.epochs, args.tokens, args.lr, args.hours) <= 0:
        p.error('Training budgets must be positive')
    if args.tokens > 32:
        p.error('This bounded experiment supports at most 32 continuation tokens')
    if not args.upload_prefix.startswith('experiments/') or '..' in args.upload_prefix:
        p.error('Use a unique experiments/... destination')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        p.error('Set CUDA_VISIBLE_DEVICES to one allocated GPU')
    args.work.mkdir(parents=True, exist_ok=True)
    lock = lock_workdir(args.work)
    out = args.work / 'outputs'
    if out.exists():
        p.error('Work directory already contains outputs; use a new directory')
    api = HfApi()
    check_destination(api, args.upload_prefix)
    info = api.dataset_info(REPO, revision=CHECKPOINT_REVISION, files_metadata=True)
    if not info.private:
        raise ValueError('Expected private parent checkpoint')
    entry = next(s for s in info.siblings if s.rfilename == CHECKPOINT)
    path = Path(hf_hub_download(REPO, CHECKPOINT, repo_type='dataset', revision=CHECKPOINT_REVISION))
    if not entry.lfs or base.file_sha(path) != entry.lfs.sha256:
        raise ValueError('Parent checkpoint checksum mismatch')
    parent = torch.load(path, map_location='cpu', weights_only=True)
    run = parent['run']
    for key, filename in [('training_script_sha256', 'train_cache_attention.py'),
                          ('architecture_script_sha256', 'cache_attention.py')]:
        if run[key] != base.file_sha(Path(__file__).with_name(filename)):
            raise ValueError('Parent checkpoint code identity mismatch')
    args.source, args.target, args.batch = run['source'], run['target'], run['batch']
    data = Dataset(SimpleNamespace(local_data=None, data_root=args.data_root, pilot=False))
    if (data.manifest_sha != run['manifest_sha256'] or
            [s['id'] for s in data.train] != run['train_ids'] or
            [s['id'] for s in data.validation] != run['validation_ids']):
        raise ValueError('Parent checkpoint dataset mismatch')
    train = data.train[:8] if args.pilot else data.train
    validation = data.validation[:4] if args.pilot else data.validation
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    mapper = CacheAttention(run['config']).to(args.device)
    mapper.load_state_dict(parent['model'])
    spec = base.MODELS[args.target]
    receiver = AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'],
                dtype=torch.float32, attn_implementation='eager').eval().to(args.device)
    receiver.requires_grad_(False)
    eos = receiver.generation_config.eos_token_id
    args.eos_ids = set(eos if isinstance(eos, list) else [eos])
    out.mkdir()
    provenance = {'parent_revision': CHECKPOINT_REVISION, 'parent_checkpoint': CHECKPOINT,
        'parent_sha256': entry.lfs.sha256, 'parent_run': run, 'tokens': args.tokens,
        'handoff_prompt_tokens': 1, 'epochs': args.epochs, 'lr': args.lr, 'seed': args.seed,
        'train_ids': [s['id'] for s in train], 'validation_ids': [s['id'] for s in validation],
        'script_sha256': base.file_sha(Path(__file__)), 'pilot': args.pilot,
        'objective': 'native-continuation teacher-to-student KL + 0.01 standardized prefix KV MSE',
        'evaluation_scope': 'Development validation already used for parent selection; not a final test'}
    base.write_json(out / 'run.json', provenance)
    traces = {}
    with torch.no_grad():
        for sample in train + validation:
            ids = torch.tensor([sample['tokens'][args.target]['ids']], device=args.device)
            native = native_start(receiver, ids)
            control = cached_start(receiver, data.features(sample, args.target)[:-1].to(args.device), ids[:, -1:])
            if not torch.allclose(native.logits, control.logits, rtol=1e-3, atol=1e-3):
                raise ValueError('Native one-token replay control failed')
            trace = continue_greedy(receiver, native, args.tokens, args.eos_ids)
            if trace != continue_greedy(receiver, control, args.tokens, args.eos_ids):
                raise ValueError('Native continuation control failed')
            traces[sample['id']] = trace
    base.write_json(out / 'native-traces.json', traces)
    before = score(mapper, receiver, data, validation, traces, args, greedy=True)
    base.write_json(out / 'before.json', before)
    train_before = score(mapper, receiver, data, train[:8], traces, args, greedy=True)
    base.write_json(out / 'train-before.json', train_before)
    print('BEFORE: '+json.dumps(before['summary']), flush=True)
    best = before['summary']['kl']
    checkpoint(out / 'best.pt', {'model': mapper.state_dict(), 'run': run, 'calibration': provenance})
    upload_outputs(args)
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=args.lr, weight_decay=1e-4)
    steps, stopped = 0, False
    for epoch in range(args.epochs):
        order = list(train)
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for sample in order:
            if time.perf_counter() - started >= args.hours * 3600:
                stopped = True
                break
            mapper.train()
            ids = torch.tensor([sample['tokens'][args.target]['ids']], device=args.device)
            source, target = data.features(sample, args.source), data.features(sample, args.target)
            teacher = teacher_logits(receiver, ids, traces[sample['id']])
            _, suffix = continuation_inputs(ids, traces[sample['id']])
            optimizer.zero_grad(set_to_none=True)
            mapped = translate_prefix(mapper, source, sample, args, ids.shape[1] - 1)
            student = receiver_logits(receiver, mapped, suffix)
            _, valid = causal_positions(sample, args.source, args.target)
            mask = valid[:-1].to(args.device)
            anchor = F.mse_loss(mapper.standardize_target(mapped)[mask],
                               mapper.standardize_target(target[:-1].to(args.device))[mask])
            loss = logit_kl(student, teacher) + .01 * anchor
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite calibration loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
            steps += 1
            if steps % 10 == 0:
                print(f'Epoch {epoch+1}/{args.epochs}, prompt {len(losses)}/{len(train)}: {loss.item():.5f}', flush=True)
            if steps % 25 == 0:
                checkpoint(out / 'last.pt', {'model': mapper.state_dict(), 'run': run,
                           'calibration': provenance, 'steps': steps})
                upload_outputs(args)
        validation_score = score(mapper, receiver, data, validation, traces, args)
        if validation_score['summary']['kl'] < best:
            best = validation_score['summary']['kl']
            checkpoint(out / 'best.pt', {'model': mapper.state_dict(), 'run': run, 'calibration': provenance})
        row = {'epoch': epoch+1, 'steps': steps, 'prompts_seen_this_epoch': len(losses),
               'training_loss': sum(losses)/len(losses) if losses else None,
               'validation': validation_score['summary'], 'elapsed_seconds': time.perf_counter()-started}
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        checkpoint(out / 'last.pt', {'model': mapper.state_dict(), 'run': run,
                   'calibration': provenance, 'steps': steps})
        upload_outputs(args)
        if stopped:
            break
    mapper.load_state_dict(torch.load(out / 'best.pt', map_location=args.device, weights_only=True)['model'])
    after = score(mapper, receiver, data, validation, traces, args, greedy=True)
    train_after = score(mapper, receiver, data, train[:8], traces, args, greedy=True)
    result = {'before': before, 'after': after, 'train_before': train_before,
              'train_after': train_after, 'steps': steps, 'budget_reached': stopped,
              'wall_seconds': time.perf_counter()-started,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
    base.write_json(out / 'comparison.json', result)
    upload_outputs(args)
    print('COMPLETE: '+json.dumps({'before': before['summary'], 'after': after['summary']}), flush=True)
    lock.close()


if __name__ == '__main__':
    main()
