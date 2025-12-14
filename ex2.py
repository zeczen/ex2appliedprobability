#Students Eyal Seckbach 324863539 Nitzan Davari 301733408

import argparse
import re
import sys
import logging
from collections import Counter,defaultdict
from math import log2
import time


def compute_and_output29_table(train_half, heldout_half, V, best_lam,logger):
    CT = Counter(train_half)
    CH = Counter(heldout_half)

    T_size = len(train_half)
    H_size = len(heldout_half)

    # r -> list of words
    freq_to_words = defaultdict(list)
    for w, r in CT.items():
        if r <= 9:
            freq_to_words[r].append(w)

    # r = 0 (unseen in training)
    unseen_words = set(CH.keys()) - set(CT.keys())
    freq_to_words[0] = list(unseen_words)

    for r in range(0, 10):  # EXACTLY 0..9
        words_r = freq_to_words.get(r, [])

        Nr = len(words_r)
        Tr = sum(CH[w] for w in words_r)

        # f_lambda (Lidstone expected frequency)
        p_lam = (r + best_lam) / (T_size + best_lam * V)
        f_lam = p_lam * T_size

        # f_H (held-out expected frequency)
        if r == 0:
            denom = (V - len(CT)) * H_size
        else:
            denom = Nr * H_size

        p_H = Tr / denom if denom > 0 else 0.0
        f_H = p_H * T_size

        logger.info(
            f"{r}\t"
            f"{f_lam:.5f}\t"
            f"{f_H:.5f}\t"
            f"{Nr}\t"
            f"{Tr}"
        )





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

    parser.add_argument(
        "--executionType",
        choices=["regular", "fastest", "debug"],
        default="regular",
        help="Execution mode: regular (default), fastest, or debug"
    )

    return parser


def setup_file_logger(output_path: str) -> logging.Logger:
    logger = logging.getLogger("ex2")
    # Remove existing handlers to avoid duplicate lines if function called multiple times
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output_path, encoding="utf-8", mode='w')
    logger.addHandler(handler)
    return logger


def parse_text_words(file_path: str , is_test) -> list:
    
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


def perplexity_lidstone(lam, validation_set, vocabulary, N_train, V):
    """
    Computes the perplexity of a validation set using Lidstone smoothing.

    Parameters:
    - lam: float, Lidstone smoothing parameter (λ)
    - validation_set: list of str, words in the validation set
    - vocabulary: dict, mapping words to their counts in the training set
    - N_train: int, total number of words in the training set
    - V: int, size of the vocabulary

    Returns:
    - float, perplexity of the validation set
    """

    # Initialize the sum of log probabilities
    log_sum = 0.0

    # Number of words in the validation set
    N = len(validation_set)

    # Iterate over each word in the validation set
    for word in validation_set:
        # Get the count of the word in the training vocabulary; default to 0 if unseen
        count = vocabulary.get(word, 0)

        # Compute the smoothed probability using Lidstone estimator
        prob = lidstone_estimate(count, N_train, V, lam)

        # Add the log2 of probability to the sum
        # Use max(prob, 1e-12) to avoid log(0) if probability is extremely small
        log_sum += log2(max(prob, 1e-12))

    # Compute perplexity: 2 raised to the negative average log2 probability
    return 2 ** (-log_sum / N)




def precompute_heldout_stats(train, heldout, V):
    """
    Returns:
    p_r : dict mapping r -> held-out probability
    """

    CT = Counter(train)
    CH = Counter(heldout)

    # r -> list of words
    freq_to_words = defaultdict(list)
    for w, r in CT.items():
        freq_to_words[r].append(w)

    # r = 0 (unseen in training)
    unseen_words = set(CH) - set(CT)
    freq_to_words[0] = list(unseen_words)

    p_r = {}
    H_size = len(heldout)

    for r, words in freq_to_words.items():
        Nr = len(words)

        if r == 0:
            denom = (V - len(CT)) * H_size
        else:
            denom = Nr * H_size

        Tr = sum(CH[w] for w in words)
        p_r[r] = Tr / denom if denom > 0 else 0.0

    return p_r, CT



def perplexity_heldout(validation_set, V, train, heldout):
    p_r, CT = precompute_heldout_stats(train, heldout, V)

    log_sum = 0.0
    N = len(validation_set)

    for word in validation_set:
        r = CT.get(word, 0)
        prob = p_r.get(r, 1e-12)  # safety against log(0)
        log_sum += log2(prob)

    return 2 ** (-log_sum / N)



def held_out(V,train, heldout,input_word):
    
    train_frequencies = Counter(train)
    heldout_frequencies = Counter(heldout)

    input_freq = 0
    if input_word in  train_frequencies :
        input_freq = train_frequencies[input_word]

    Nr_words = []
    Nr =0
    if input_freq != 0:
        #collecting all words with same frequency from trainig
        for word in train_frequencies:
            if train_frequencies[word] == input_freq:
                Nr_words.append(word)
                Nr+=1
    else: #handling unseen word case - get all the words that exist in held out and unexist in train
        Nr_words = set(heldout_frequencies) - set(train_frequencies)
        Nr = V - len(train_frequencies)



    Tr = 0
    for word in Nr_words:
        Tr+= heldout_frequencies[word]
    
    return float(Tr) / (Nr * len(heldout))



    """
    words               = list of training tokens
    vocabulary          = list or set of all V possible tokens
    lam                 = Lidstone lambda
    lidstone_estimate   = lidstone_estimate(count, S, V, lam)
    held_out            = your held-out function: held_out(V, train, heldout, input_word)
    """
def debug_models(words, V, lam):


    wordsOc = Counter(words)          # word -> count
    S = len(words)                    # total tokens in training set

    # seen words
    seen_words = list(wordsOc.keys())

    # unseen words count
    n0 = V - len(seen_words)

    if n0 < 0:
        raise ValueError("Vocabulary smaller than number of unique words?")

    #  HELPER: run model function
    def get_prob(model, word):
        """
        model = "lidstone" or "heldout"
        word  = input token (string)
        """
        count = wordsOc.get(word, 0)

        if model == "lidstone":
            return lidstone_estimate(count, S, V, lam)

        elif model == "heldout":
            # held_out(V, train_counts, heldout_counts, word)
            heldout_words = ["bla","blalba","dummyword"]
            return held_out(V, wordsOc, heldout_words, word)

        else:
            raise ValueError("Invalid model type")

    # Debug each model
    for modelName in ["lidstone", "heldout"]:

        # p(x*) — probability for one unseen word
        p_unseen_single = get_prob(modelName, "__UNSEEN__")   # count=0

        # probability mass for unseen block
        unseen_mass = p_unseen_single * n0

        # probability mass for all seen words
        seen_mass = sum(get_prob(modelName, w) for w in seen_words)

        check_sum = unseen_mass + seen_mass

        print(f"{modelName.upper()}  -> mass_seen={seen_mass:.6f}, mass_unseen={unseen_mass:.6f}, total={check_sum:.6f}")

        if abs(check_sum - 1.0) > 1e-5:
            raise RuntimeError(f"ERROR: {modelName} probabilities do NOT sum to 1 (sum={check_sum})")
        else:
            print(f"{modelName} test PASSED ✓\n")

    print("All checks complete.")



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
    logger.info(f"#Students\tEyal\tSeckbach\t324863539\tNitzan\tDavari\t301733408")
    logger.info(f"#Output1\t{args.development_set}")
    logger.info(f"#Output2\t{args.test_set}")
    logger.info(f"#Output3\t{args.input_word}")
    logger.info(f"#Output4\t{args.output_file}")
    logger.info(f"#Output5\t{V}")
    logger.info(f"#Output6\t{1 / V}")

    # 2. Development set preprocessing
    #train_set = parse_text(args.development_set)
    train_set_words = parse_text_words(args.development_set,False)
    logger.info(f"#Output7\t{len(train_set_words)}")

    # 3. Lidstone model training
    validation_set = train_set_words[round(len(train_set_words) * 0.9):]
    train_set = train_set_words[:round(len(train_set_words) * 0.9)]
    logger.info(f"#Output8\t{len(validation_set)}")
    logger.info(f"#Output9\t{len(train_set)}")

    vocabulary = Counter(word for s in train_set for word in s.split())
    #V = len(vocabulary)
    #N = sum(vocabulary.values())
    N = len(train_set)
    logger.info(f"#Output10\t{len(vocabulary)}")
    logger.info(f"#Output11\t{vocabulary[args.input_word]}")

    logger.info(f"#Output12\t{maximum_likelihood_estimate(vocabulary[args.input_word], N)}")
    logger.info(f"#Output13\t{maximum_likelihood_estimate(vocabulary['unseen-word'], N)}")

    logger.info(f"#Output14\t{lidstone_estimate(vocabulary[args.input_word], N, V, 0.1)}")
    logger.info(f"#Output15\t{lidstone_estimate(vocabulary['unseen-word'], N, V, 0.1)}")

    
    end = time.perf_counter()
    print(f"before perplexities :: Execution time: {end - start:.6f} seconds")

    perplexities = {}
    perplexities_range = range (1,200)
    if args.executionType == "fastest":
        print("fast execution mode on - reducing perplexity calculations")
        perplexities_range = [1,6,10,100]

    for i in perplexities_range:
        lam = i / 100
        perplexities[lam] = perplexity_lidstone(lam, validation_set, vocabulary, N, V)


    
    end = time.perf_counter()
    print(f"after perplexities :: Execution time: {end - start:.6f} seconds")

    logger.info(f"#Output16\t{perplexities[0.01]}")
    logger.info(f"#Output17\t{perplexities[0.1]}")
    logger.info(f"#Output18\t{perplexities[1]}")
    min_lam = min(perplexities, key=perplexities.get)
    logger.info(f"#Output19\t{min_lam}")
    logger.info(f"#Output20\t{min(perplexities.values())}")

    # 4. Held out model training
    first_halve_training = train_set_words[:round(len(train_set_words) * 0.5)]
    second_halve_heldout = train_set_words[round(len(train_set_words) * 0.5):]
    
    logger.info(f"#Output21\t{len(first_halve_training)}")
    logger.info(f"#Output22\t{len(second_halve_heldout)}")
    
    logger.info(f"#Output23\t{held_out(V,first_halve_training, second_halve_heldout,args.input_word)}")
    logger.info(f"#Output24\t{held_out(V,first_halve_training, second_halve_heldout,'unseen-word')}")

    
    
    end = time.perf_counter()
    print(f"after heldout before debugging :: Execution time: {end - start:.6f} seconds")
    print(f"execution type : {args.executionType}")
    #5 debugging modules:
    if args.executionType == "debug":
        debug_models(train_set_words,V ,lam=0.1)


        
    end = time.perf_counter()
    print(f"after debugging :: Execution time: {end - start:.6f} seconds")


    test_words = parse_text_words(args.test_set,True)
    test_N = len(test_words)
    logger.info(f"#Output25\t{len(test_words)}")

    
    print(f"after parse_text_words  for test, before perplexity_lidstone  time: {end - start:.6f} seconds")

    test_set_perplexity = perplexity_lidstone(min_lam, test_words, vocabulary, test_N, V)
    logger.info(f"#Output26\t{test_set_perplexity}")


    print(f" before perplexity_heldout  time: {end - start:.6f} seconds")
    test_perplexity_heldout_value = perplexity_heldout(test_words, V, first_halve_training, second_halve_heldout)
    logger.info(f"#Output27\t{test_perplexity_heldout_value}")

    if test_set_perplexity < test_perplexity_heldout_value:
        logger.info(f"#Output28\tL")
    else:
        logger.info(f"#Output28\tH")


    print(f" before output29 table  time: {end - start:.6f} seconds")

    logger.info(f"#Output29\t")
        # Output29
    compute_and_output29_table(
        first_halve_training,
        second_halve_heldout,
        V,
        min_lam,
        logger
    )


    end = time.perf_counter()
    print(f"after all :: Execution time: {end - start:.6f} seconds")




if __name__ == "__main__":
    main()
