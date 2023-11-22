import datetime
import random
import re
import string

import jieba
import torch
import torch.nn.functional as F
from antllm.utils import mpu
from antllm.utils.generation_utils import (BeamSearchScorer,
                                           LogitsProcessorList,
                                           MinLengthLogitsProcessor,
                                           NoRepeatNGramLogitsProcessor)
from antllm.utils.logging.logger import log_dist
from tqdm import tqdm


def _is_digit(w):
    for ch in w:
        if not (ch.isdigit() or ch == ","):
            return False
    return True


gigaword_tok_dict = {
    "(": "-lrb-",
    ")": "-rrb-",
    "[": "-lsb-",
    "]": "-rsb-",
    "{": "-lcb-",
    "}": "-rcb-",
    "[UNK]": "UNK",
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
}

cnndm_tok_dict = {
    "(": "-LRB-",
    ")": "-RRB-",
    "[": "-LSB-",
    "]": "-RSB-",
    "{": "-LCB-",
    "}": "-RCB-",
}


def fix_tokenization(text, dataset):
    if dataset == "cnn_dm_org":
        return text
    if dataset == "gigaword":
        text = text.replace("[UNK]", "UNK")
        return text
    input_tokens = text.split()
    output_tokens = []
    has_left_quote = False
    has_left_single_quote = False

    i = 0
    prev_dash = False
    while i < len(input_tokens):
        tok = input_tokens[i]
        flag_prev_dash = False
        if tok == '"':
            if has_left_quote:
                output_tokens.append("''")
            else:
                output_tokens.append("``")
            has_left_quote = not has_left_quote
            i += 1
        elif (
            tok == "'"
            and len(output_tokens) > 0
            and output_tokens[-1].endswith("n")
            and i < len(input_tokens) - 1
            and input_tokens[i + 1] == "t"
        ):
            output_tokens[-1] = output_tokens[-1][:-1]
            output_tokens.append("n't")
            i += 2
        elif (
            tok == "'"
            and i < len(input_tokens) - 1
            and input_tokens[i + 1] in ("s", "d", "ll")
        ):
            output_tokens.append("'" + input_tokens[i + 1])
            i += 2
        elif tok == "'":
            if has_left_single_quote:
                output_tokens.append("'")
            else:
                output_tokens.append("`")
            has_left_single_quote = not has_left_single_quote
            i += 1
        elif (
            tok == "."
            and i < len(input_tokens) - 2
            and input_tokens[i + 1] == "."
            and input_tokens[i + 2] == "."
        ):
            output_tokens.append("...")
            i += 3
        elif (
            tok == ","
            and len(output_tokens) > 0
            and _is_digit(output_tokens[-1])
            and i < len(input_tokens) - 1
            and _is_digit(input_tokens[i + 1])
        ):
            # $ 3 , 000 -> $ 3,000
            output_tokens[-1] += "," + input_tokens[i + 1]
            i += 2
        elif (
            tok == "."
            and len(output_tokens) > 0
            and output_tokens[-1].isdigit()
            and i < len(input_tokens) - 1
            and input_tokens[i + 1].isdigit()
        ):
            # 3 . 03 -> $ 3.03
            output_tokens[-1] += "." + input_tokens[i + 1]
            i += 2
        elif (
            tok == "."
            and len(output_tokens) > 0
            and len(output_tokens[-1]) == 1
            and output_tokens[-1].isalpha()
            and i < len(input_tokens) - 2
            and len(input_tokens[i + 1]) == 1
            and input_tokens[i + 1].isalpha()
            and input_tokens[i + 2] == "."
        ):
            # U . N . -> U.N.
            k = i + 3
            while k + 2 < len(input_tokens):
                if (
                    len(input_tokens[k + 1]) == 1
                    and input_tokens[k + 1].isalpha()
                    and input_tokens[k + 2] == "."
                ):
                    k += 2
                else:
                    break
            output_tokens[-1] += "".join(input_tokens[i:k])
            i = k
        elif tok == "-":
            if i < len(input_tokens) - 1 and input_tokens[i + 1] == "-":
                output_tokens.append("--")
                i += 2
            elif i == len(input_tokens) - 1 or i == 0:
                output_tokens.append("-")
                i += 1
            elif (
                output_tokens[-1] not in string.punctuation
                and input_tokens[i + 1][0] not in string.punctuation
            ):
                output_tokens[-1] += "-"
                i += 1
                flag_prev_dash = True
            else:
                output_tokens.append("-")
                i += 1
        elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
            output_tokens[-1] += tok
            i += 1
        else:
            output_tokens.append(tok)
            i += 1
        prev_dash = flag_prev_dash
    return " ".join(output_tokens)


def count_tokens(tokens):
    counter = {}
    for t in tokens:
        if t in counter.keys():
            counter[t] += 1
        else:
            counter[t] = 1
    return counter


def get_f1(text_a, text_b):
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    if len(tokens_a) == 0 or len(tokens_b) == 0:
        return 1 if len(tokens_a) == len(tokens_b) else 0
    set_a = count_tokens(tokens_a)
    set_b = count_tokens(tokens_b)
    match = 0
    for token in set_a.keys():
        if token in set_b.keys():
            match += min(set_a[token], set_b[token])
    p = match / len(tokens_a)
    r = match / len(tokens_b)
    return 2.0 * p * r / (p + r + 1e-5)


def remove_duplicate(l_list, duplicate_rate):
    tk_list = [x.lower().split() for x in l_list]
    r_list = []
    history_set = set()
    for i, w_list in enumerate(tk_list):
        w_set = set(w_list)
        if len(w_set) > 0 and len(w_set & history_set) / len(w_set) <= duplicate_rate:
            r_list.append(l_list[i])
        history_set |= w_set
    return r_list


def rouge_metric(
    predictions,
    labels,
    examples,
    metric="rouge-1",
    duplicate_rate=0.7,
    dataset="cnn_dm",
):
    # metric_dict = {"rouge-1": "rouge1", "rouge-2": "rouge2", "rouge-l": "rougeLsum"}
    refs = [example.meta["ref"] for example in examples]
    ref_list = []
    for ref in refs:
        ref = ref.strip().split("[SEP]")
        ref = [fix_tokenization(sentence, dataset=dataset) for sentence in ref]
        ref = [" ".join(jieba.lcut(r)) for r in ref]
        ref = "\n".join(ref)
        ref_list.append(ref)
    pred_list = []
    for prediction in predictions:
        buf = []
        for sentence in prediction.strip().split("[SEP]"):
            sentence = fix_tokenization(sentence, dataset=dataset)
            if any(get_f1(sentence, s) > 1.0 for s in buf):
                continue
            # comment it since it will cause empty pred.
            # s_len = len(sentence.split())
            # if s_len <= 4:
            #     continue
            buf.append(sentence)
        if duplicate_rate and duplicate_rate < 1:
            buf = remove_duplicate(buf, duplicate_rate)
        buf = [" ".join(jieba.lcut(r)) for r in buf]
        line = "\n".join(buf)
        pred_list.append(line)
    if torch.distributed.get_rank() == 0:
        import json

        with open("./results.json", "w") as output:
            for ref, pred in zip(ref_list, pred_list):
                output.write(
                    json.dumps({"ref": ref, "pred": pred}, ensure_ascii=False) + "\n"
                )
    #    scorer = rouge_scorer.RougeScorer([metric_dict[metric]], use_stemmer=True)
    #    scores = [scorer.score(pred, ref) for pred, ref in zip(pred_list, ref_list)]
    #    scores = [score[metric_dict[metric]].fmeasure for score in scores]
    #    scores = sum(scores) / len(scores)
    from rouge_metric import ROUGE

    rouge = ROUGE(metrics=[metric])
    metrics = rouge.compute_score(ref_list, pred_list)
    scores = metrics[metric.upper()]["F"]
    return scores


def squad_fix_tokenization(text):
    text = text.replace(" - ", "-")
    text = text.replace(" \u2013 ", "\u2013")
    text = re.sub(r"\ba \. m \.\b", "a.m.", text)
    text = re.sub(r"\ba \. m\b", "a.m", text)
    text = re.sub(r"\bp \. m \.\b", "p.m.", text)
    text = re.sub(r"\bp \. m\b", "p.m", text)
    text = re.sub(r"\b' s\b", "'s", text)
    text = re.sub(r"\bu \. s \.\b", "u.s.", text)
    text = re.sub(r"\bu \. s\b", "u.s", text)
    tokens = text.split()
    i = 0
    while i < len(tokens):
        if tokens[i] in [",", ".", ":"] and i > 0 and i + 1 < len(tokens):
            if tokens[i - 1][-1].isdigit() and tokens[i + 1][0].isdigit():
                if tokens[i] == "," and len(tokens[i + 1]) > 3:
                    pass
                else:
                    tokens[i - 1] = tokens[i - 1] + tokens[i] + tokens[i + 1]
                    tokens = tokens[:i] + tokens[i + 2:]
                    i -= 1
        i += 1
    text = " ".join(tokens)
    return text


def squad_decode(example, prediction, tokenizer):
    text = tokenizer.decode(prediction)
    if text.replace(" ", "").lower() == "n/a":
        return text
    context = example.meta["context"]
    context_tokens = example.meta["context_tokens"]
    token_to_char = example.meta["token_to_char"]
    for i in range(len(context_tokens)):
        if prediction == context_tokens[i: i + len(prediction)]:
            s = token_to_char[i][0]
            t = token_to_char[i + len(prediction) - 1][1]
            return context[s:t]
    text = squad_fix_tokenization(text)
    return text


# remove punctuation
sp_char = [
    "-",
    ":",
    "_",
    "*",
    "^",
    "/",
    "\\",
    "~",
    "`",
    "+",
    "=",
    "，",
    "。",
    "：",
    "？",
    "！",
    "“",
    "”",
    "；",
    "’",
    "《",
    "》",
    "……",
    "·",
    "、",
    ",",
    "。",
    "(",
    ")",
    "「",
    "」",
    "（",
    "）",
    "－",
    "～",
    "『",
    "』",
]


def remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return "".join(out_segs)


unk = "⁇"


def fix_unk(context_raw, pred_raw):
    if unk not in pred_raw:
        return pred_raw
    pred0 = pred_raw.replace(f" {unk} ", unk, 1111)
    pred0 = pred0.replace(f"{unk} ", unk, 1111)
    pred0 = pred0.replace(f" {unk}", unk, 1111)
    context = remove_punctuation(context_raw)
    while pred0.find(unk) >= 0:
        idx = pred0.find(unk)
        left = ""
        if idx > 0:
            left = remove_punctuation(pred0[:idx])
            start = context.find(left)
            if start < 0:
                import pdb
                pdb.set_trace()
            else:
                start = start + len(left)
                word = context[start]
                pred0 = pred0[:idx] + word + pred0[idx + 1:]
                continue
        if idx + 1 < len(pred0):
            right = remove_punctuation(pred0[idx + 1:])
            end = context.find(right)
            if end < 0:
                import pdb
                pdb.set_trace()
                # xx = 1
            else:
                word = context[end - 1]
                pred0 = pred0[:idx] + word + pred0[idx + 1:]
                continue
        break

    if unk in pred0:
        #        import pdb;pdb.set_trace()
        return pred_raw
    print(f"{pred_raw} => {pred0}")
    return pred0


def cmrc_decode(example, prediction, tokenizer):
    text = tokenizer.decode(prediction)
    if text.replace(" ", "").lower() == "n/a":
        return text
    #    import pdb;pdb.set_trace()
    context = example.meta["context"]
    context_tokens = example.meta["context_tokens"]
    token_to_char = example.meta["token_to_char"]
    for i in range(len(context_tokens)):
        if prediction == context_tokens[i: i + len(prediction)]:
            s = token_to_char[i][0]
            t = token_to_char[i + len(prediction) - 1][1]
            return fix_unk(context, context[s:t])
    text = squad_fix_tokenization(text)
    return fix_unk(context, text)


def process_batch(batch, args):
    """Process batch and produce inputs for the model."""
    if "mask" in batch:
        # finetune SQuAD
        batch["attention_mask"] = batch.pop("mask")
        batch["position_id"] = batch.pop("position")
    tokens = batch["text"].long().cuda()
    attention_mask = batch["attention_mask"].long().cuda()
    position_ids = batch["position_id"].long().cuda()
    if tokens.dim() == 3:
        tokens = tokens.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        position_ids = position_ids.squeeze(1)
    return tokens, attention_mask, position_ids


class DecoderEvaluater:
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.start_token = tokenizer.sop_token_id
        self.end_token = tokenizer.eop_token_id
        self.mask_token = (
            tokenizer.smask_token_id
            if args.task_mask and args.task != "cmrc"
            else tokenizer.mask_token_id
        )
        self.pad_token = tokenizer.pad_token_id
        self.processors = LogitsProcessorList()
        if args.min_tgt_length > 0:
            processor = MinLengthLogitsProcessor(args.min_tgt_length, self.end_token)
            self.processors.append(processor)
        if args.no_repeat_ngram_size > 0:
            processor = NoRepeatNGramLogitsProcessor(args.no_repeat_ngram_size)
            self.processors.append(processor)

    def evaluate(self, model, dataloader, example_dict, args):
        """Calculate correct over total answers and return prediction if the
        `output_predictions` is true."""
        model.eval()
        local_predictions = {}
        log_dist("Distributed store created")
        with torch.no_grad():
            # For all the batches in the dataset.
            for idx, data in tqdm(enumerate(dataloader)):
                if idx >= args.max_eval_steps:
                    break
                tokens, attention_mask, position_ids = process_batch(data, args)
                batch_size = tokens.size(0)
                beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    max_length=args.out_seq_length,
                    num_beams=args.num_beams,
                    device=tokens.device,
                    length_penalty=args.length_penalty,
                    do_early_stopping=False,
                )
                beam_scores = torch.zeros(
                    (batch_size, args.num_beams),
                    dtype=torch.float,
                    device=tokens.device,
                )
                beam_scores[:, 1:] = -1e9
                beam_scores = beam_scores.view((batch_size * args.num_beams,))
                # Run the model forward.
                counter = 0
                while counter < args.tgt_seq_length:
                    if counter == 0:
                        next_token_logits, *mems = model(
                            tokens, position_ids, attention_mask, return_memory=True
                        )
                        seq_length = next_token_logits.size(
                            1
                        )  # [bsz, seq_len, vocab_size]
                        next_token_logits = next_token_logits[
                            :, -1
                        ]  # [bsz, vocab_size]
                        next_token_logits = (
                            next_token_logits.unsqueeze(1)
                            .repeat(1, args.num_beams, 1)
                            .view(batch_size * args.num_beams, -1)
                        )  # [bsz*beam_size, vocab_size]
                        mems = [
                            mem.unsqueeze(1)
                            .repeat(1, args.num_beams, 1, 1)
                            .view(batch_size * args.num_beams, seq_length, -1)
                            for mem in mems
                        ]
                        # mems: [B, beam, seq_len, vocab_size]
                        position_ids = tokens.new_ones(batch_size, args.num_beams, 2, 1)
                        for i, text in enumerate(tokens.tolist()):
                            mask_pos = text.index(self.mask_token)
                            position_ids[i, :, 0] = mask_pos
                        position_ids = position_ids.reshape(
                            batch_size * args.num_beams, 2, 1
                        )
                        tokens = tokens.new_zeros(batch_size * args.num_beams, 0)
                        attention_mask = tokens.new_zeros([batch_size * args.num_beams])
                    else:
                        if not args.no_block_position:
                            position_ids[:, 1] = counter + 1
                        last_token = tokens[:, -1:]
                        next_token_logits, *mems = model(
                            last_token,
                            position_ids,
                            attention_mask,
                            *mems,
                            return_memory=True,
                        )
                        next_token_logits = next_token_logits[:, -1]
                    next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                    next_token_scores = self.processors(tokens, next_token_scores)
                    next_token_scores = next_token_scores + beam_scores[
                        :, None
                    ].expand_as(next_token_scores)
                    vocab_size = next_token_scores.shape[-1]
                    next_token_scores = next_token_scores.view(
                        batch_size, args.num_beams * vocab_size
                    )

                    probs = F.softmax(next_token_scores, dim=-1)
                    if args.select_topk:
                        _, next_tokens = torch.topk(
                            probs, k=2 * args.num_beams, dim=-1, largest=True
                        )
                    else:
                        next_tokens = torch.multinomial(
                            probs, num_samples=2 * args.num_beams
                        )
                    next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                    next_token_scores, _indices = torch.sort(
                        next_token_scores, descending=True, dim=1
                    )
                    next_tokens = torch.gather(next_tokens, -1, _indices)

                    next_indices = next_tokens // vocab_size
                    next_tokens = next_tokens % vocab_size
                    # stateless
                    beam_outputs = beam_scorer.process(
                        tokens,
                        next_token_scores,
                        next_tokens,
                        next_indices,
                        eos_token_id=self.end_token,
                        pad_token_id=self.pad_token,
                    )
                    beam_scores = beam_outputs["next_beam_scores"]
                    beam_next_tokens = beam_outputs["next_beam_tokens"]
                    beam_idx = beam_outputs["next_beam_indices"]
                    beam_next_tokens = beam_next_tokens.unsqueeze(-1)
                    tokens = torch.cat([tokens[beam_idx, :], beam_next_tokens], dim=-1)
                    mems = [mem[beam_idx] for mem in mems] if mems else []
                    if beam_scorer.is_done:
                        break
                    counter += 1
                tokens, _, scores = beam_scorer.finalize(
                    tokens,
                    beam_scores,
                    next_tokens,
                    next_indices,
                    eos_token_id=self.end_token,
                    pad_token_id=self.pad_token,
                )
                uid_list = data["uid"]
                if isinstance(uid_list, torch.Tensor):
                    uid_list = uid_list.cpu().numpy().tolist()
                predictions = []
                for i, text in enumerate(tokens.tolist()):
                    text = [
                        token
                        for token in text
                        if token not in [self.end_token, self.pad_token]
                    ]
                    if args.task in [
                        "squad",
                        "squad_v1",
                    ] and args.tokenizer_model_type.startswith("bert"):
                        uid = uid_list[i]
                        example = example_dict[uid]
                        text = squad_decode(example, text, self.tokenizer)
                    elif args.task == "cmrc":
                        uid = uid_list[i]
                        example = example_dict[uid]
                        text = cmrc_decode(example, text, self.tokenizer)

                    else:
                        text = self.tokenizer.decode(text)
                    predictions.append(text)
                for uid, prediction in zip(uid_list, predictions):
                    local_predictions[uid] = prediction
                if (idx + 1) % args.log_interval == 0:
                    log_dist(f"Iteration {idx + 1} / {len(dataloader)}")
        model.train()
        torch.distributed.barrier()
        log_dist("Evaluation completed")
        gathered_predictions = [None for i in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_predictions, local_predictions)
        gathered_predictions = {
            uid: pred for preds in gathered_predictions for uid, pred in preds.items()
        }
        predictions, examples = [], []
        for uid, example in example_dict.items():
            if uid not in gathered_predictions:
                continue
            prediction = gathered_predictions[uid]
            predictions.append(prediction)
            examples.append(example)
        torch.distributed.barrier()
        return predictions, [], examples


def blanklm_fix_tokenization(text):
    text = text.replace("` `", "``")
    text = text.replace("' '", "''")
    text = text.replace("n ' t", "n't")
    text = text.replace("' s", "'s")
    text = text.replace("' m", "'m")
    text = text.replace("' re", "'re")
    text = text.replace(". . .", "...")
    text = text.replace(" . .", " ..")
    text = text.replace("- -", "--")
    text = text.replace("u . s .", "u.s.")
    text = text.replace("u . k .", "u.k.")
    text = text.replace("e . g .", "e.g.")
    return text


class BlankLMEvaluater(DecoderEvaluater):
    def evaluate(self, model, dataloader, example_dict, args):
        model.eval()
        store = torch.distributed.TCPStore(
            args.master_ip,
            18931 + random.randint(0, 10000),
            mpu.get_data_parallel_world_size(),
            torch.distributed.get_rank() == 0,
            datetime.timedelta(seconds=30),
        )
        log_dist("Distributed store created")

        with torch.no_grad():
            for idx, data in enumerate(dataloader):
                tokens, attention_mask, position_ids = process_batch(data, args)
                src_tokens = tokens
                batch_size = tokens.size(0)
                mask_positions = []
                current_mask = []
                for text in tokens.tolist():
                    mask_positions.append(
                        [i for i, x in enumerate(text) if x == self.mask_token]
                    )
                    current_mask.append(0)
                    # print(self.tokenizer.decode(text))
                    # print(mask_positions[-1])
                counter = 0
                done = [False] * batch_size
                while counter < args.tgt_seq_length:
                    if counter == 0:
                        # print(tokens)
                        # print(position_ids)
                        next_token_logits, *mems = model(
                            tokens, position_ids, attention_mask, return_memory=True
                        )
                        next_token_logits = next_token_logits[:, -1]
                        position_ids = tokens.new_ones(batch_size, 2, 1)
                        for i, text in enumerate(tokens.tolist()):
                            mask_pos = mask_positions[i][current_mask[i]]
                            position_ids[i, 0] = mask_pos
                        tokens = tokens.new_zeros(batch_size, 0)
                        attention_mask = tokens.new_zeros(batch_size)
                    else:
                        position_ids[:, 1] = position_ids[:, 1] + 1
                        last_token = tokens[:, -1:]
                        next_token_logits, *mems = model(
                            last_token,
                            position_ids,
                            attention_mask,
                            *mems,
                            return_memory=True,
                        )
                        next_token_logits = next_token_logits[:, -1]
                    next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                    next_token_scores = self.processors(tokens, next_token_scores)
                    next_tokens = next_token_scores.max(dim=-1)[1]
                    # print(self.tokenizer.decode(next_tokens.tolist()))
                    for i, next_token in enumerate(next_tokens.tolist()):
                        if next_token == self.end_token:
                            if current_mask[i] + 1 < len(mask_positions[i]):
                                current_mask[i] += 1
                                next_tokens[i] = self.start_token
                                position_ids[i, 0] = mask_positions[i][current_mask[i]]
                                position_ids[i, 1] = 0
                            else:
                                done[i] = True
                        if done[i]:
                            next_tokens[i] = self.pad_token
                    if all(done):
                        break
                    tokens = torch.cat([tokens, next_tokens.unsqueeze(-1)], dim=-1)
                    counter += 1
                predictions = []
                for i, text in enumerate(tokens.tolist()):
                    text = [
                        token
                        for token in text
                        if token not in [self.end_token, self.pad_token]
                    ]
                    blanks = [[]]
                    for token in text:
                        if token == self.start_token:
                            blanks.append([])
                        else:
                            blanks[-1].append(token)
                    output_tokens = []
                    current_blank = 0
                    for token in src_tokens[i].tolist():
                        if token == self.mask_token:
                            if current_blank < len(blanks):
                                output_tokens += blanks[current_blank]
                            current_blank += 1
                        else:
                            if token not in [self.pad_token]:
                                output_tokens.append(token)
                    text = self.tokenizer.decode(output_tokens[:-1])
                    text = blanklm_fix_tokenization(text)
                    predictions.append(text)
                    # print(text)
                uid_list = data["uid"]
                if isinstance(uid_list, torch.Tensor):
                    uid_list = uid_list.cpu().numpy().tolist()
                for uid, prediction in zip(uid_list, predictions):
                    store.set(uid, prediction)
                if (idx + 1) % args.log_interval == 0:
                    log_dist(f"Iteration {idx + 1} / {len(dataloader)}")

        model.train()
        torch.distributed.barrier()
        log_dist("Evaluation completed")
        predictions, examples = [], []
        for uid, example in example_dict.items():
            predictions.append(store.get(uid).decode("utf-8"))
            examples.append(example)
        torch.distributed.barrier()
        return predictions, [], examples
