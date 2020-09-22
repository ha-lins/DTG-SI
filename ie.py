#!/usr/bin/env python3
"""
Information Retrieval.
Equivalent to the following commands:

cd data2text
GOLD_FILE="../nba_data/gold.valid.txt"
GEN_FILE="../attn_copynet_sd_path/ckpt/hypos.step96860.val.txt"
INTER_FILE="${GEN_FILE}.h5"
GPUID=1
python data_utils.py prep_gen_data --dict_pfx roto-ie --input_path rotowire -val_file ${GOLD_FILE} --gen ${GEN_FILE} --output ${INTER_FILE}
th extractor.lua -gpuid ${GPUID} -datafile roto-ie.h5 -dict_pfx roto-ie -just_eval -preddata ${INTER_FILE}
python non_rg_metrics.py ${GOLD_FILE} ${INTER_FILE}-tuples.txt
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import subprocess
from data2text.data_utils import prep_generated_data, get_json_dataset
from data2text.non_rg_metrics import calc_precrec, get_items
from utils import corpus_bleu, read_sents_from_file

data2text_dir = "data2text"


def get_step_number(filename, verbose=False):
    basename = os.path.basename(filename)
    basename_parts = basename.split(".")
    try:
        return int(basename_parts[1][len("step"):])
    except ValueError:
        print("Cannot extract step number in {}".format(basename))
        return 0


def get_bleu(ref_file, hypo_file):
    refs = read_sents_from_file(ref_file)
    refs = [[ref] for ref in refs]
    hypos = read_sents_from_file(hypo_file)
    bleu = corpus_bleu(refs, hypos) * 1e-2
    print("BLEU: {:.2%}".format(bleu))
    return bleu


def get_precrec(
        gold_file, hypo_file, inter_file, ref_file, bleu_only=False, gpuid=0,
        dict_pfx=os.path.join(data2text_dir, "roto-ie"),
        json_path=os.path.join(data2text_dir, "rotowire"),
        write_record=True):
    if not bleu_only:
        rough_res = prep_generated_data(
            hypo_file, dict_pfx, inter_file,
            trdata=get_json_dataset(json_path, "train"),
            val_file=gold_file)

        ret = subprocess.call(
            ["th", "extractor.lua",
             "-gpuid", str(gpuid+1),
             "-datafile", os.path.abspath("{}.h5".format(dict_pfx)),
             "-dict_pfx", os.path.abspath(dict_pfx),
             "-just_eval",
             "-preddata", os.path.abspath(inter_file)],
            cwd=data2text_dir)
        if ret != 0:
            raise Exception(
                "run extractor.lua failed with return value {}".format(ret))

        pred_file = "{}-tuples.txt".format(inter_file)
        gold_items, pred_items = map(get_items, (gold_file, pred_file))
        with open("{}.res.txt".format(hypo_file), "w") as itemwise_outfile:
            precrec = calc_precrec(gold_items, pred_items,
                                   itemwise_outfile=itemwise_outfile)

    else:
        rough_res = 0., 0.
        precrec = 0., 0.

    bleu = get_bleu(ref_file, hypo_file)

    if write_record:
        dirname, basename = os.path.split(hypo_file)
        basename_parts = basename.split(".")
        step = get_step_number(hypo_file)
        stage = basename_parts[-2]
        with open(os.path.join(dirname, "ie_results.{}.txt".format(stage)), 'a') as results_file:
            print("{}\t{}".format(step, "\t".join(map("{:.6f}".format, (bleu,) + precrec + rough_res))), file=results_file)
            results_file.flush()

    return precrec


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--gold_file", default=os.path.join("nba_data", "gold.test.txt"))
    argparser.add_argument("--ref_file", default=os.path.join("nba_data", "nba.sent_ref.test.txt"))
    argparser.add_argument("--bleu_only", action="store_true")
    argparser.add_argument("hypo_files", nargs="+")
    argparser.add_argument("--inter_file", default="")
    argparser.add_argument("--gpuid", type=int, default=0)
    args = argparser.parse_args()
    args.hypo_files.sort(key=lambda hypo_file: (get_step_number(hypo_file, verbose=True), hypo_file))
    for hypo_file in args.hypo_files:
        print("processing {}:".format(hypo_file))
        if not args.inter_file:
            s = hypo_file
            suffix = ".txt"
            if s.endswith(suffix):
                s = s[:-len(suffix)]
            args.inter_file = s + ".h5"
        prec, rec = get_precrec(
            args.gold_file, hypo_file, args.inter_file, args.ref_file, bleu_only=args.bleu_only, gpuid=args.gpuid)
