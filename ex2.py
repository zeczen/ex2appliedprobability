# Students: Eyal Seckbach 324863539, Nitzan Davari 301733408
# Exercise: Applied Probability - Exercise 2
# This script computes word probability models (Lidstone and Held-Out),
# evaluates perplexity on a validation/test set, and outputs frequency tables.

import argparse
import re
import sys
import logging
from collections import Counter, defaultdict
from math import log2
import time


def compute_and_output29_table(train_half, heldout_half, V, best_lam, len_train_set, logger):
    """
    Computes and logs a table of expected frequencies for words based on their
    training frequencies (r=0..9) comparing Lidstone vs Held-Out models.
    
    Parameters:
    - train_half: list of str, first half of training words
    - heldout_half: list of str, held-out portion of training words
    - V: int, vocabulary size
    - best_lam: float, best Lidstone lambda determined by validation
    - len_train_set: int, total number of words in the training set
    - logger: logging.Logger, to write outputs
    """
    train_counters = Counter(train_half)
    heldout_counters = Counter(heldout_half)

    len_train = len(train_half)
    H_size = len(heldout_half)

    # Map frequency r -> list of words in training with that frequency
    freq_to_words = defaultdict(list)
    for w, r in train_counters.items():
        if r <= 9:  # only consider r=1..9
            freq_to_words[r].append(w)

    # r = 0: words unseen in training but appear in held-out
    unseen_words = set(heldout_counters.keys()) - set(train_counters.keys())
    freq_to_words[0] = list(unseen_words)

    for r in range(10):  # for r = 0..9
        words_r = freq_to_words.get(r, [])
        Nr = len(words_r)
        if r == 0:  # special case for unseen words
            Nr = V - len(set(train_counters.keys()))
        Tr = sum(heldout_counters[w] for w in words_r)  # total held-out occurrences

        # f_lambda: expected frequency using Lidstone smoothing
        f_lam = lidstone_estimate(r, len_train_set, V, best_lam) * len_train_set

        # f_H: expected frequency using held-out model
        denom = (V - len(train_counters)) * H_size if r == 0 else Nr * H_size
        p_H = Tr / denom if denom > 0 else 0.0
        f_H = p_H * len_train

        # Log output as required by assignment
        logger.info(
            f"{r}\t"
            f"{f_lam:.5f}\t"
            f"{f_H:.5f}\t"
            f"{Nr}\t"
            f"{Tr}"
        )


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="ex2.py",
        description="Exercise 2 of applied probability"
    )
    parser.add_argument("development_set", type=str, help="Path to development set file")
    parser.add_argument("test_set", type=str, help="Path to test set file")
    parser.add_argument("input_word", type=str, help="Word to check probabilities")
    parser.add_argument("output_file", type=str, help="Output log file path")
    parser.add_argument(
        "--executionType",
        choices=["regular", "fastest", "debug"],
        default="regular",
        help="Execution mode: regular, fastest (reduce computations), or debug"
    )
    return parser


def setup_file_logger(output_path: str) -> logging.Logger:
    """
    Set up a file logger to save program outputs in UTF-8 encoding.
    """
    logger = logging.getLogger("ex2")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output_path, encoding="utf-8", mode='w')
    logger.addHandler(handler)
    return logger


def parse_text_words(file_path: str, is_test: bool) -> list:
    """
    Parse words from a TRAIN or TEST file.
    
    Args:
    - file_path: path to the text file
    - is_test: whether the file is a test set
    Returns: list of words (tokens)
    """
    header_re = re.compile(r'<TRAIN\s+(\d+)[^>]*>', re.IGNORECASE)
    if is_test:
        header_re = re.compile(r'<TEST\s+(\d+)[^>]*>', re.IGNORECASE)

    with open(file_path, "r", encoding="utf-8") as fh:
        text = fh.read()

    matches = list(header_re.finditer(text))
    words = []

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        for word in block.split():
            words.append(word)

    return words


def maximum_likelihood_estimate(count_word: int, total_words: int) -> float:
    """
    MLE of a word: count / total words
    """
    if total_words == 0:
        return 0.0
    return count_word / total_words


def lidstone_estimate(count_word: int, S: int, V: int, lam: float) -> float:
    """
    Lidstone smoothed probability for a word.
    P(word) = (count + λ) / (total words + λ * V)
    """
    if S == 0:
        return 0.0
    return (count_word + lam) / (S + lam * V)


def perplexity_lidstone(lam, validation_set, vocabulary, N_train, V):
    """
    Compute perplexity for a validation set using Lidstone smoothing.
    Lower perplexity -> better model fit.
    """
    log_sum = 0.0
    N = len(validation_set)
    for word in validation_set:
        count = vocabulary.get(word, 0)
        prob = lidstone_estimate(count, N_train, V, lam)
        log_sum += log2(max(prob, 1e-12))  # avoid log(0)
    return 2 ** (-log_sum / N)


def precompute_heldout_stats(train, heldout, V):
    """
    Precompute held-out probabilities p_r for words with training frequency r.
    Returns:
    - p_r: dict mapping r -> held-out probability
    - CT: Counter of training words
    """
    CT = Counter(train)
    CH = Counter(heldout)

    freq_to_words = defaultdict(list)
    for w, r in CT.items():
        freq_to_words[r].append(w)

    # r=0: words unseen in training
    unseen_words = set(CH) - set(CT)
    freq_to_words[0] = list(unseen_words)

    p_r = {}
    H_size = len(heldout)

    for r, words in freq_to_words.items():
        Nr = len(words)
        denom = (V - len(CT)) * H_size if r == 0 else Nr * H_size
        Tr = sum(CH[w] for w in words)
        p_r[r] = Tr / denom if denom > 0 else 0.0

    return p_r, CT


def perplexity_heldout(validation_set, V, train, heldout):
    """
    Compute held-out model perplexity.
    """
    p_r, CT = precompute_heldout_stats(train, heldout, V)
    log_sum = 0.0
    N = len(validation_set)
    for word in validation_set:
        r = CT.get(word, 0)
        prob = p_r.get(r, 1e-12)  # avoid log(0)
        log_sum += log2(prob)
    return 2 ** (-log_sum / N)


def held_out(V, train, heldout, input_word):
    """
    Compute held-out probability for a specific input word.
    """
    train_frequencies = Counter(train)
    heldout_frequencies = Counter(heldout)

    input_freq = train_frequencies.get(input_word, 0)

    Nr_words = []
    Nr = 0
    if input_freq != 0:
        # all words in training with same frequency
        for word in train_frequencies:
            if train_frequencies[word] == input_freq:
                Nr_words.append(word)
                Nr += 1
    else:
        # unseen word: words in heldout not in training
        Nr_words = set(heldout_frequencies) - set(train_frequencies)
        Nr = V - len(train_frequencies)

    Tr = sum(heldout_frequencies[word] for word in Nr_words)
    return float(Tr) / (Nr * len(heldout))


def debug_models(words, V, lam):
    """
    Check that Lidstone and Held-Out models probabilities sum to 1.
    Raises RuntimeError if sum differs significantly from 1.
    """
    wordsOc = Counter(words)
    S = len(words)
    seen_words = list(wordsOc.keys())
    n0 = V - len(seen_words)
    if n0 < 0:
        raise ValueError("Vocabulary smaller than number of unique words?")

    def get_prob(model, word):
        count = wordsOc.get(word, 0)
        if model == "lidstone":
            return lidstone_estimate(count, S, V, lam)
        elif model == "heldout":
            # dummy heldout list for check
            heldout_words = ["bla", "blalba", "dummyword"]
            return held_out(V, wordsOc, heldout_words, word)
        else:
            raise ValueError("Invalid model type")

    for modelName in ["lidstone", "heldout"]:
        p_unseen_single = get_prob(modelName, "__UNSEEN__")
        unseen_mass = p_unseen_single * n0
        seen_mass = sum(get_prob(modelName, w) for w in seen_words)
        check_sum = unseen_mass + seen_mass

        print(f"{modelName.upper()}  -> mass_seen={seen_mass:.6f}, mass_unseen={unseen_mass:.6f}, total={check_sum:.6f}")
        if abs(check_sum - 1.0) > 1e-5:
            raise RuntimeError(f"ERROR: {modelName} probabilities do NOT sum to 1")
        else:
            print(f"{modelName} test PASSED ✓\n")

    print("All checks complete.")


def main(argv=None):
    """
    Main execution function:
    - parses args
    - computes Lidstone and Held-Out models
    - evaluates perplexity on validation/test sets
    - logs results and frequency tables
    """
    start = time.perf_counter()
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_file_logger(args.output_file)

    V = 300_000  # fixed vocabulary size

    # Log input information
    logger.info(f"#Student\tEyal Seckbach\t324863539\tNitzan Davari\t301733408")
    logger.info(f"#Output1\t{args.development_set}")
    logger.info(f"#Output2\t{args.test_set}")
    logger.info(f"#Output3\t{args.input_word}")
    logger.info(f"#Output4\t{args.output_file}")
    logger.info(f"#Output5\t{V}")
    logger.info(f"#Output6\t{1 / V}")

    # 1. Parse development set
    train_set_words = parse_text_words(args.development_set, False)
    logger.info(f"#Output7\t{len(train_set_words)}")

    # 2. Split train/validation (90/10)
    validation_set = train_set_words[round(len(train_set_words) * 0.9):]
    train_set = train_set_words[:round(len(train_set_words) * 0.9)]
    logger.info(f"#Output8\t{len(validation_set)}")
    logger.info(f"#Output9\t{len(train_set)}")

    vocabulary_train_set = Counter(word for s in train_set for word in s.split())
    N = len(train_set)
    logger.info(f"#Output10\t{len(vocabulary_train_set)}")
    logger.info(f"#Output11\t{vocabulary_train_set[args.input_word]}")
    logger.info(f"#Output12\t{maximum_likelihood_estimate(vocabulary_train_set[args.input_word], N)}")
    logger.info(f"#Output13\t{maximum_likelihood_estimate(vocabulary_train_set['unseen-word'], N)}")
    logger.info(f"#Output14\t{lidstone_estimate(vocabulary_train_set[args.input_word], N, V, 0.1)}")
    logger.info(f"#Output15\t{lidstone_estimate(vocabulary_train_set['unseen-word'], N, V, 0.1)}")

    # 3. Compute perplexities for various Lidstone λ
    perplexities = {}
    perplexities_range = range(0, 200)
    if args.executionType == "fastest":
        perplexities_range = [1, 6, 10, 100]

    for i in perplexities_range:
        lam = i / 100
        perplexities[lam] = perplexity_lidstone(lam, validation_set, vocabulary_train_set, N, V)

    logger.info(f"#Output16\t{perplexities[0.01]}")
    logger.info(f"#Output17\t{perplexities[0.1]}")
    logger.info(f"#Output18\t{perplexities[1]}")
    min_lam = min(perplexities, key=perplexities.get)
    logger.info(f"#Output19\t{min_lam}")
    logger.info(f"#Output20\t{min(perplexities.values())}")

    # 4. Held-out model
    first_halve_training = train_set_words[:round(len(train_set_words) * 0.5)]
    second_halve_heldout = train_set_words[round(len(train_set_words) * 0.5):]
    logger.info(f"#Output21\t{len(first_halve_training)}")
    logger.info(f"#Output22\t{len(second_halve_heldout)}")
    logger.info(f"#Output23\t{held_out(V, first_halve_training, second_halve_heldout, args.input_word)}")
    logger.info(f"#Output24\t{held_out(V, first_halve_training, second_halve_heldout, 'unseen-word')}")

    if args.executionType == "debug":
        debug_models(train_set_words, V, lam=0.1)

    # 5. Test set evaluation
    test_words = parse_text_words(args.test_set, True)
    test_set_perplexity = perplexity_lidstone(min_lam, test_words, vocabulary_train_set, N, V)
    test_perplexity_heldout_value = perplexity_heldout(test_words, V, first_halve_training, second_halve_heldout)
    logger.info(f"#Output25\t{len(test_words)}")
    logger.info(f"#Output26\t{test_set_perplexity}")
    logger.info(f"#Output27\t{test_perplexity_heldout_value}")
    logger.info(f"#Output28\t{'L' if test_set_perplexity < test_perplexity_heldout_value else 'H'}")

    # 6. Compute output29 table comparing Lidstone vs Held-out expected frequencies
    logger.info(f"#Output29")
    compute_and_output29_table(first_halve_training, second_halve_heldout, V, min_lam, len(train_set), logger)

    end = time.perf_counter()
    print(f"Execution complete. Total time: {end - start:.6f} seconds")


if __name__ == "__main__":
    main()
