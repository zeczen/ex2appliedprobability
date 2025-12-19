# Students Eyal Seckbach 324863539

import argparse
import re
import sys
import logging
import time
from collections import Counter, defaultdict
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


def parse_text_words(file_path: str, is_test) -> list:
    # regex: match '<TRAIN' whitespace digits ... '>'    (case-insensitive)
    header_re = re.compile(r'<TRAIN\s+(\d+)[^>]*>', re.IGNORECASE)
    if is_test:
        header_re = re.compile(r'<TEST\s+(\d+)[^>]*>', re.IGNORECASE)

    with open(file_path, "r", encoding="utf-8") as fh:
        text = fh.read()

    matches = list(header_re.finditer(text))
    ids = []
    data = []
    words = []

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        rec_id = int(m.group(1))
        ids.append(rec_id)
        for word in block.split():
            words.append(word)

    return words


def maximum_likelihood_estimate(count_word: int, total_words: int) -> float:
    """Calculate the maximum likelihood estimate (MLE) of a word given its count and the total word count."""
    if total_words == 0:
        return 0.0
    return count_word / total_words


def lidstone_estimate(count_word: int, S: int, V: int, lam: float) -> float:
    """Calculate the Lidstone estimate of a word given its count, the # total word, vocabulary size, and smoothing parameter lam.
        N: total number of words in the training set
        V: vocabulary size, total of unique words in the training set
    """
    if S == 0:
        return 0.0
    return (count_word + lam) / (S + lam * V)


def perplexity_lidstone(lam, val_voc, vocabulary, V):
    """ Calculate the perplexity of a test set using Lidstone smoothing."""
    return 2 ** (
            -sum(
                log2(lidstone_estimate(vocabulary[word], vocabulary.total(), V, lam)) * c
                for word, c in val_voc.items()
            ) / val_voc.total()
    )


def perplexity_heldout(val_voc, V, train_voc, heldout_voc, p_heldout):
    """ Calculate the perplexity of a test set using Held-out estimation."""
    return 2 ** (
        (-sum(
            log2(p_heldout[train_voc[word]]) * c
            for word, c in val_voc.items()
        )
         / val_voc.total())
    )


def heldout(train_voc, heldout_voc, V):
    """
    Calculate the held-out probability estimates.
    train_voc : dict of tokens with their frequency in training
    heldout_voc : dict of tokens with their frequency in held-out set
    V              : vocabulary size
    """
    # r : frequency in training -> list of words with that frequency
    r = {freq: [w for w, f in train_voc.items() if f == freq]
         for freq in set(train_voc.values())}

    prob_ho = defaultdict(float)
    H = heldout_voc.total()


    for f, words in r.items():
        Nr = len(words)
        Tr = sum(heldout_voc[w] for w in words)
        prob_ho[f] = Tr / (Nr * H)

    N0 = r[0]
    T0 = sum(heldout_voc[w] for w in N0)
    prob_ho[0] = T0 / (max(V - sum(1 for w, c in train_voc.items() if c != 0), 0) * H)

    return prob_ho


def main(argv=None):
    start = time.perf_counter()

    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_file_logger(args.output_file)

    V = 300_000
    # At this point args.development_set, args.test_set, args.input_word,
    # and args.output_file are available as strings.

    # 1. Init
    logger.info(f"#Students\tEyal\tSeckbach\t324863539")
    logger.info(f"#Output1\t{args.development_set}")
    logger.info(f"#Output2\t{args.test_set}")
    logger.info(f"#Output3\t{args.input_word}")
    logger.info(f"#Output4\t{args.output_file}")
    logger.info(f"#Output5\t{V}")
    logger.info(f"#Output6\t{1 / V}")

    # 2. Development set preprocessing
    # train_set = parse_text(args.development_set)
    train_set = parse_text_words(args.development_set, False)

    logger.info(f"#Output7\t{len(train_set)}")

    # 3. Lidstone model training
    validation_set = train_set[round(len(train_set) * 0.9):]
    first_halve_training = train_set[:round(len(train_set) * 0.5)]
    second_halve_heldout = train_set[round(len(train_set) * 0.5):]
    train_set = train_set[:round(len(train_set) * 0.9)]
    test_set = parse_text_words(args.test_set, True)

    N = len(train_set)

    logger.info(f"#Output8\t{len(validation_set)}")
    logger.info(f"#Output9\t{len(train_set)}")

    vocabulary = Counter(train_set)
    val_voc = Counter(validation_set)
    test_voc = Counter(test_set)
    # V = len(vocabulary)
    # N = sum(vocabulary.values())
    logger.info(f"#Output10\t{len(vocabulary)}")
    logger.info(f"#Output11\t{vocabulary[args.input_word]}")

    logger.info(f"#Output12\t{maximum_likelihood_estimate(vocabulary[args.input_word], N)}")
    logger.info(f"#Output13\t{maximum_likelihood_estimate(vocabulary['unseen-word'], N)}")

    logger.info(f"#Output14\t{lidstone_estimate(vocabulary[args.input_word], N, V, 0.1)}")
    logger.info(f"#Output15\t{lidstone_estimate(vocabulary['unseen-word'], N, V, 0.1)}")

    perplexities = {}
    for lam in [0.01, 0.1, 0.06, 1]:
        perplexities[lam] = perplexity_lidstone(lam, val_voc, vocabulary, V)

    logger.info(f"#Output16\t{perplexities[0.01]}")
    logger.info(f"#Output17\t{perplexities[0.1]}")
    logger.info(f"#Output18\t{perplexities[1]}")
    min_lam = min(perplexities, key=perplexities.get)
    logger.info(f"#Output19\t{min_lam}")
    logger.info(f"#Output20\t{min(perplexities.values())}")

    # 4. Held out model training
    f_voc = Counter(first_halve_training)
    s_voc = Counter(second_halve_heldout)
    for w in s_voc.keys() - f_voc.keys():
        f_voc[w] = 0


    logger.info(f"#Output21\t{len(first_halve_training)}")
    logger.info(f"#Output22\t{len(second_halve_heldout)}")

    p_ho = heldout(f_voc, s_voc, V)  # Precompute heldout probabilities

    logger.info(f"#Output23\t{p_ho[f_voc[args.input_word]]}")
    logger.info(f"#Output24\t{p_ho[f_voc['unseen-word']]}")

    logger.info(f"#Output25\t{len(test_set)}")

    test_set_perplexity = perplexity_lidstone(min_lam, test_voc, vocabulary, V)
    logger.info(f"#Output26\t{test_set_perplexity}")

    test_perplexity_heldout_value = perplexity_heldout(test_voc, V, f_voc, s_voc, p_ho)
    logger.info(f"#Output27\t{test_perplexity_heldout_value}")

    if test_set_perplexity < test_perplexity_heldout_value:
        logger.info(f"#Output28\tL")
    else:
        logger.info(f"#Output28\tH")

    logger.info(f"#Output29")
    for r in range(10):
        f_lam = N * lidstone_estimate(r, N, V, min_lam)
        f_h = f_voc.total() * p_ho[r]
        Ntr = sum(1 for c in f_voc.values() if c == r)
        Tr = sum(c for w, c in s_voc.items() if f_voc[w] == r)

        logger.info(f"{r}\t{f_lam:.5f}\t{f_h:.5f}\t{V - len(set(first_halve_training)) if r == 0 else Ntr}\t{Tr}")

    print(f"Exec time: {time.perf_counter() - start:.6f} seconds")


if __name__ == "__main__":
    main()
