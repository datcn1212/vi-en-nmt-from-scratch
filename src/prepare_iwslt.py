"""Preprocessing for the IWSLT15 en-vi: HTML unescape, NFC
normalisation, and a length filter applied to the train split.
"""
import argparse
import html
import os
import unicodedata


def clean_lines(path):
    with open(path, encoding="utf-8") as f:
        return [unicodedata.normalize("NFC", html.unescape(line.rstrip("\n"))) for line in f]


def write_lines(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def filter_by_length(src_lines, tgt_lines, max_words):
    keep = [i for i in range(len(src_lines))
            if len(src_lines[i].split()) <= max_words
            and len(tgt_lines[i].split()) <= max_words]
    dropped = len(src_lines) - len(keep)
    print(f"train: dropped {dropped}/{len(src_lines)} pairs longer than {max_words} words "
          f"({100 * dropped / len(src_lines):.2f}%)")
    return [src_lines[i] for i in keep], [tgt_lines[i] for i in keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_words", type=int, required=True,
                         help="drop train pairs where either side has more whitespace words "
                              "than this - pick from the corpus's own word-length percentiles, "
                              "not a copied default")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    splits = [("train", "train"), ("tst2012", "dev"), ("tst2013", "test")]
    for raw_name, out_name in splits:
        vi_lines = clean_lines(os.path.join(args.raw_dir, f"{raw_name}.vi"))
        en_lines = clean_lines(os.path.join(args.raw_dir, f"{raw_name}.en"))
        assert len(vi_lines) == len(en_lines), f"{raw_name}: line count mismatch"

        if out_name == "train":
            vi_lines, en_lines = filter_by_length(vi_lines, en_lines, args.max_words)

        write_lines(vi_lines, os.path.join(args.out_dir, f"{out_name}.vi"))
        write_lines(en_lines, os.path.join(args.out_dir, f"{out_name}.en"))
        print(f"{out_name}: {len(vi_lines)} pairs written")


if __name__ == "__main__":
    main()
