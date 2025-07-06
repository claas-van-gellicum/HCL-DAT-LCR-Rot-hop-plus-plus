# Method for obtaining the raw data of the three domains (laptop, restaurant, book)
#
# https://github.com/Johan-Verschoor/CL-XD-ABSA/
#
# Adapted from Knoester, Frasincar, and Trușcă (2022)
# https://doi.org/10.1007/978-3-031-20891-1_3
#
# Knoester, J., Frasincar, F., and Trușcă, M. M. (2022). Domain adversarial training for aspect-
# based sentiment analysis. In 22nd International Conference on Web Information Systems
# Engineering (WISE 2022), volume 13724 of LNCS, pages 21–37. Springer.


import os
import nltk
import argparse
from data_book_hotel import read_book_hotel
from data_rest_lapt import read_rest_lapt


def main():
    """
    Gets the raw data for the specified domain.

    :return:
    """
    # Domain is one of the following: restaurant (2014), laptop (2014), book (2019).
    # Ensure the punkt tokenizer is available

    # CLI argument parsing
    parser = argparse.ArgumentParser(description="Extract raw data for ABSA domains.")
    parser.add_argument("--domain", type=str, required=True, help="Domain to process: restaurant, laptop, or book")
    parser.add_argument("--year", type=int, required=True, help="Year of the dataset (2014 or 2019 (book))")
    args = parser.parse_args()

    nltk.download('punkt')

    
    domain = args.domain
    year = args.year

    if domain in ["restaurant", "laptop"]:
        train_file = f"data/externalData/{domain}_train_{year}.xml"
        test_file = f"data/externalData/{domain}_test_{year}.xml"
        train_out = f"data/programGeneratedData/BERT/{domain}/raw_data_{domain}_train_{year}.txt"
        test_out = f"data/programGeneratedData/BERT/{domain}/raw_data_{domain}_test_{year}.txt"

        os.makedirs(os.path.dirname(train_out), exist_ok=True)
        os.makedirs(os.path.dirname(test_out), exist_ok=True)

        with open(train_out, "w") as out:
            out.write("")
        with open(test_out, "w") as out:
            out.write("")

        read_rest_lapt(
            in_file=train_file,
            source_count=[], source_word2idx={},
            target_count=[], target_phrase2idx={},
            out_file=train_out
        )

        read_rest_lapt(
            in_file=test_file,
            source_count=[], source_word2idx={},
            target_count=[], target_phrase2idx={},
            out_file=test_out
        )

    elif domain == "book":
        in_file = f"data/externalData/{domain}_reviews_{year}.xml"
        out_file = f"data/programGeneratedData/BERT/{domain}/raw_data_{domain}_{year}.txt"

        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        with open(out_file, "w") as out:
            out.write("")

        read_book_hotel(
            in_file=in_file,
            source_count=[], source_word2idx={},
            target_count=[], target_phrase2idx={},
            out_file=out_file
        )

    else:
        raise ValueError(f"Unsupported domain: {domain}. Choose from: restaurant, laptop, book")


if __name__ == '__main__':
    main()