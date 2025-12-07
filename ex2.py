import argparse
import re
import sys
import logging
from collections import Counter,defaultdict
from math import log2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ex2.py",
        description="Exercise 2 of applied probability"
    )
    parser.add_argument(
        "development_set",
        type=str,
        help="Path to the development set filename (string)."
    )

    parser.add_argument(
        "test_set",
        type=str,
        help="Path to the test set filename (string)."
    )
    parser.add_argument(
        "input_word",
        type=str,
        help="INPUT WORD (string)."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output filename (string)."
    )

    return parser


def setup_file_logger(output_path: str) -> logging.Logger:
    logger = logging.getLogger("ex2")
    # Remove existing handlers to avoid duplicate lines if function called multiple times
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output_path, encoding="utf-8", mode='w')
    logger.addHandler(handler)
    return logger


def parse_text(file_path: str) -> list:
    # regex: match '<TRAIN' whitespace digits ... '>'    (case-insensitive)
    header_re = re.compile(r'<TRAIN\s+(\d+)[^>]*>', re.IGNORECASE)

    with open(file_path, "r", encoding="utf-8") as fh:
        text = fh.read()

    matches = list(header_re.finditer(text))
    ids = []
    data = []

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        rec_id = int(m.group(1))
        ids.append(rec_id)
        data.append(block)

    return data


def maximum_likelihood_estimate(count_word: int, total_words: int) -> float:
    """Calculate the maximum likelihood estimate (MLE) of a word given its count and the total word count."""
    if total_words == 0:
        return 0.0
    return count_word / total_words


def lidstone_estimate(count_word: int, N: int, V: int, lam: float) -> float:
    """Calculate the Lidstone estimate of a word given its count, the # total word, vocabulary size, and smoothing parameter lam.
        N: total number of words in the training set
        V: vocabulary size, total of unique words in the training set
    """
    if N == 0:
        return 0.0
    return (count_word + lam) / (N + lam * V)


def perplexity(lam, validation_set, vocabulary, N, V) -> float:
    """Calculate the perplexity of the Lidstone model with lambda lam on the validation set."""

    return 2 ** ((-1 / N) * sum(log2(
        lidstone_estimate(vocabulary.get(word, 0), N, V, lam)
    ) for s in validation_set for word in s.split()))

def get_sorted_freq_histogram(workingset)-> dict:
    word_counts = Counter(word for s in workingset for word in s.split())
    set = set(word_counts.keys())
    freq_hist = defaultdict(int)
    for word in set:
        freq_hist[word_counts[word]] += 1
    return dict(sorted(freq_hist.items()))


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_file_logger(args.output_file)

    V = 300_000
    # At this point args.development_set, args.test_set, args.input_word,
    # and args.output_file are available as strings.

    # 1. Init
    logger.info(f"Students Eyal Seckbach 324863539 Nitzan Davari 301733408")
    logger.info(f"Output1:\t{args.development_set}")
    logger.info(f"Output2:\t{args.test_set}")
    logger.info(f"Output3:\t{args.input_word}")
    logger.info(f"Output4:\t{args.output_file}")
    logger.info(f"Output5:\t{V}")
    logger.info(f"Output6:\t{1 / V}")

    # 2. Development set preprocessing
    train_set = parse_text(args.development_set)
    logger.info(f"Output7:\t{len(train_set)}")

    # 3. Lidstone model training
    validation_set = train_set[round(len(train_set) * 0.9):]
    train_set = train_set[:round(len(train_set) * 0.9)]
    logger.info(f"Output8:\t{len(validation_set)}")
    logger.info(f"Output9:\t{len(train_set)}")

    vocabulary = Counter(word for s in train_set for word in s.split())
    V = len(vocabulary)
    N = sum(vocabulary.values())
    logger.info(f"Output10:\t{V}")
    logger.info(f"Output11:\t{vocabulary[args.input_word]}")

    logger.info(f"Output12:\t{maximum_likelihood_estimate(vocabulary[args.input_word], N)}")
    logger.info(f"Output13:\t{maximum_likelihood_estimate(vocabulary['unseen-word'], N)}")

    logger.info(f"Output14:\t{lidstone_estimate(vocabulary[args.input_word], N, V, 0.1)}")
    logger.info(f"Output15:\t{lidstone_estimate(vocabulary['unseen-word'], N, V, 0.1)}")

    perplexities = {0.01: perplexity(0.01, validation_set, vocabulary, N, V),
                    0.1: perplexity(0.1, validation_set, vocabulary, N, V),
                    1.0: perplexity(1.0, validation_set, vocabulary, N, V)}

    logger.info(f"Output16:\t{perplexities[0.01]}")
    logger.info(f"Output17:\t{perplexities[0.1]}")
    logger.info(f"Output18:\t{perplexities[1]}")
    logger.info(f"Output19:\t{min(perplexities, key=perplexities.get)}")
    logger.info(f"Output20:\t{min(perplexities.values())}")

    # 4. Held out model training
    first_halve_training = train_set[round(len(train_set) * 0.5):]
    second_halve_heldout = train_set[:round(len(train_set) * 0.5)]
    logger.info(f"Output21:\t{len(first_halve_training)}")
    logger.info(f"Output22:\t{len(second_halve_heldout)}")

    
    train_freq_hist_sorted = get_sorted_freq_histogram (first_halve_training )
    heldout_freq_hist_sorted = get_sorted_freq_histogram (second_halve_heldout )




    

    vocabulary_heldout = Counter(word for s in second_halve_heldout for word in s.split())
    held_out_occurences = vocabulary_heldout[args.input_word]





if __name__ == "__main__":
    main()
