"""Processing data for pretraining."""

import os
import random
import sys
import time
import argparse
import json
import multiprocessing
import numpy as np
from glob import glob
from antllm.models.glm.tokenization_glm import GLMTokenizer

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)


class DynamicArray:
    def __init__(self, name, size, capacity, dtype):
        self.name = name
        self.dtype = dtype
        self.max_val = np.iinfo(self.dtype).max
        self.data = np.zeros((capacity,), dtype=self.dtype)
        self.capacity = capacity
        self.size = size

    def add(self, x):
        assert x <= self.max_val, x
        if self.size == self.capacity:
            self.capacity *= 2
            newdata = np.zeros((self.capacity,), dtype=self.dtype)
            newdata[: self.size] = self.data
            self.data = newdata
            print(f"expand {self.name} capacity to {self.capacity}")

        self.data[self.size] = x
        self.size += 1

    def __len__(self):
        return self.size

    def finalize(self):
        self.data = self.data[: self.size]
        print(f"finalize {self.name} capacity: {self.data.shape}")
        return self.data


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    filtered_sources = [
        "Github",
        "StackExchange",
        "DM Mathematics",
        "Ubuntu IRC",
        "EuroParl",
        "YoutubeSubtitles",
        "Enron Emails",
    ]
    downsample_sources = {"PubMed Central": 0.5, "FreeLaw": 0.5}

    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = GLMTokenizer.from_pretrained(self.args.spm_tokenizer_path)

    def encode_json(self, line):
        try:
            text = json.loads(line)["content"]
        except Exception as e:
            print("fail\n", line, e)
            return {}, 0

        ids = self.tokenizer.encode(text)
        ids = np.array(ids, dtype=np.int32)
        return {"text": ids}, len(text)

    def encode_text(self, line):
        text = line

        ids = self.tokenizer.encode(text)
        ids = np.array(ids, dtype=np.int32)
        return {"text": ids}, len(text)

    def encode(self, json_line):
        if self.args.type in ["json"]:
            return self.encode_json(json_line)
        elif self.args.type in ["text"]:
            return self.encode_text(json_line)
        raise NotImplementedError(f"data type {self.args.type} not support.")


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument("--input", type=str, required=True, help="Path to input JSON")
    group.add_argument(
        "--output_dir", type=str, required=True, help="Path to input JSON"
    )
    group.add_argument(
        "--json-keys",
        nargs="+",
        default=["text"],
        help="space separate listed of keys to extract from json",
    )
    group.add_argument(
        "--split-sentences", action="store_true", help="Split documents into sentences."
    )
    group.add_argument(
        "--keep-newlines",
        action="store_true",
        help="Keep newlines between sentences when splitting.",
    )

    group = parser.add_argument_group(title="tokenizer")

    group.add_argument(
        "--merge-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )
    group.add_argument(
        "--append-eod",
        action="store_true",
        help="Append an <eod> token to the end of a document.",
    )

    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--type",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )

    group.add_argument(
        "--word-segment",
        action="store_true",
        help="Append an <eod> token to the end of a document.",
    )

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers",
        type=int,
        required=True,
        help="Number of worker processes to launch",
    )
    group.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="Chunk size assigned to each worker process",
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Interval between progress updates",
    )
    group.add_argument(
        "--spm-tokenizer-path",
        type=str,
        required=True,
        help="Interval between progress updates",
    )

    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def main():
    args = get_args()
    startup_start = time.time()

    if os.path.isfile(args.input):
        read_next = open(args.input, "r", encoding="utf-8")
        print("Opening", args.input)
    else:

        def read_next0():
            for fname in glob(args.input):
                print(f"Openinng {fname}")
                with open(fname, encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        yield line

        read_next = read_next0()
    # dname = os.path.join(args.output_dir, os.path.basename(args.input) + ".text")
    # print(f"Writing to {dname}")
    # fout = open(dname, "wb")

    encoder = Encoder(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, read_next, args.chunk_size)
    # encoded_docs = map(encoder.encode, fin)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    loader_scatter = 32
    print("Time to startup:", startup_end - startup_start)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for i in range(loader_scatter):
        sfname = f"{args.output_dir}/{i}.lazy/"
        if not os.path.exists(sfname):
            os.mkdir(sfname)

    ilens = [
        DynamicArray(f"lens_{i}", 0, 1000000, np.int32) for i in range(loader_scatter)
    ]
    ws = [open(f"{args.output_dir}/{i}.lazy/text", "wb") for i in range(loader_scatter)]
    scatter_list = list(range(loader_scatter))
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        # if i>20000:break
        if bytes_processed <= 0:
            continue
        scatter_id = random.choices(scatter_list, k=1)[0]
        total_bytes_processed += bytes_processed
        ids = doc["text"]
        ilens[scatter_id].add(ids.shape[0])
        ws[scatter_id].write(ids.tobytes(order="C"))
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(
                f"Processed {i} documents",
                f"({i/elapsed} docs/s, {mbs} MB/s).",
                file=sys.stderr,
            )

    for w in ws:
        w.close()

    for i, dlens in enumerate(ilens):
        dlens = dlens.finalize()
        np.save(f"{args.output_dir}/{i}.lazy/text.lens", dlens)
        pdlens = np.zeros(dlens.shape[0], dtype=np.int32)
        with open(f"{args.output_dir}/{i}.lazy/prompt", "wb") as w:
            pass
        np.save(f"{args.output_dir}/{i}.lazy/prompt.lens", pdlens)


if __name__ == "__main__":
    main()
