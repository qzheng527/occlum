import os
import json
from tqdm import tqdm
from solutions.antllm.antllm.inference.glm_predictor import GLMForInference
from solutions.antllm.datachain.utils import load_jsonl

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def call_sft(model_path,
             datapath,
             outprefix,
             prompt_prefix="",
             do_sample=False,
             num_return_sequences=3,
             top_k=50,
             top_p=0.9,
             temperature=1,
             stidx=0,
             edidx=0,
             key="input",
             candkey="llm_cands",
             retlist=True):
    if not stidx and not edidx:
        outpath = f"{outprefix}.jsonl"
    else:
        outpath = f"{outprefix}_{stidx}_{edidx}.jsonl"
    
    bot = GLMForInference(model_path)

    samples = load_jsonl(datapath)
    if edidx:
        samples = samples[stidx: edidx]
    else:
        samples = samples[stidx:]
    
    with open(outpath, 'a', encoding='utf-8') as ofile:
        for sample in tqdm(samples):
            input = prompt_prefix + sample[key]
            answers = bot.generate(
                prompt=input,
                top_k=top_k, 
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences
            ).texts
            if isinstance(answers, list):
                sample[candkey] = answers
            else:
                sample[candkey] = [answers]

            if not retlist:
                sample[candkey] = sample[candkey][0]
    
        line = json.dumps(sample, ensure_ascii=False)
        ofile.write(line)
        ofile.write('\n')


if __name__ == "__main__":
    call_sft(datapath="/mnt2/hs272483/antnlp/chatgpt/data/zhuli_eval_0831.jsonl",
             model_path="/mnt2/hs272483/sft_models/med_sft_0831",
             outprefix="/mnt2/hs272483/antnlp/chatgpt/data/zhuli_eval_0831_sft0831",
             candkey="sft0831_prediction",
             do_sample=False,
             num_return_sequences=1,
             retlist=False)