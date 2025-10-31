# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import time
import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from janus.utils.profiling import JSONProfiler

def main():
    parser = argparse.ArgumentParser(description="Janus-Pro inference (understanding)")
    parser.add_argument("--profile", action="store_true", help="Enable JSON profiling output")
    parser.add_argument("--metrics-out", type=str, default="metrics.json", help="Path to write JSON metrics")
    args = parser.parse_args()

    # specify the path to the model
    model_path = "/data/yaxin/Janus/ckpt/Janus-Pro-7B"

    prof = JSONProfiler(enabled=args.profile, script="inference.py", model_path=model_path, out_path=args.metrics_out)

    # Model load
    prof.start_run(tag="model_load")
    with prof.measure("from_pretrained_and_move", reset_cuda_peak=True):
        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer
        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    prof.end_run()

    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>\nConvert the formula into latex code.",
            "images": ["images/equation.png"],
        },
        {"role": "Assistant", "content": ""},
    ]

    # Inference run
    prof.start_run(tag="inference", extra={"type": "understanding"})
    with prof.measure("preprocess", reset_cuda_peak=True):
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)

    with prof.measure("prepare_inputs_embeds"):
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    gen_start = time.perf_counter()
    with prof.measure("generate"):
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )
    gen_end = time.perf_counter()
    # Derive lengths for KV-cache estimate and throughput
    try:
        seqs = outputs.sequences  # type: ignore
    except Exception:
        seqs = outputs  # type: ignore

    prompt_len = int(prepare_inputs.attention_mask[0].sum().item())

    # When using inputs_embeds, HF returns only generated tokens in sequences.
    # Compute generated length excluding an initial BOS if present, and
    # stopping at EOS if it appears.
    try:
        seq0 = seqs[0]
        if hasattr(seq0, "tolist"):
            seq_list = seq0.tolist()
        else:
            seq_list = list(seq0)
        start_idx = 1 if (len(seq_list) > 0 and seq_list[0] == tokenizer.bos_token_id) else 0
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in seq_list:
            end_idx = seq_list.index(tokenizer.eos_token_id) + 1
        else:
            end_idx = len(seq_list)
        gen_len = max(0, end_idx - start_idx)
        total_len_returned = len(seq_list)
    except Exception:
        # Fallback to tensor shape if list conversion fails
        total_len_returned = int(seqs.shape[1]) if hasattr(seqs, "shape") else 0
        gen_len = total_len_returned

    elapsed = max(1e-9, gen_end - gen_start)
    tokens_per_s = gen_len / elapsed
    prof.add_metrics({
        "prompt_len": prompt_len,
        "generated_len": gen_len,
        "kv_seq_len_final": prompt_len + gen_len,
        "tokens_per_s": round(tokens_per_s, 3),
        "total_seq_len_returned": total_len_returned,
    })
    prof.end_run()

    first_seq = seqs[0] if hasattr(seqs, "__getitem__") else outputs[0]
    answer = tokenizer.decode(first_seq.cpu().tolist(), skip_special_tokens=True)
    print(f"{prepare_inputs['sft_format'][0]}", answer)

    prof.dump()


if __name__ == "__main__":
    main()
