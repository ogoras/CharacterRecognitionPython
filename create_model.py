from itertools import chain
import argparse

numbers = range(10)

uppercase_latin = range(10, 36)
uppercase_polish = range(36, 45)
uppercase = chain(uppercase_latin, uppercase_polish)

lowercase_latin = range(45, 71)
lowercase_polish = range(71, 80)
lowercase = chain(lowercase_latin, lowercase_polish)

letters = chain(uppercase, lowercase)
latin_letters = chain(uppercase_latin, lowercase_latin)
polish_letters = chain(uppercase_polish, lowercase_polish)

special_characters = range(80, 112)
whitespaces = range(112, 115)

non_letters = chain(numbers, special_characters, whitespaces)

full_dataset = range(115)

dataset_options = { "numbers": numbers,
    "uppercase_latin": uppercase_latin,
    "uppercase_polish": uppercase_polish,
    "uppercase": uppercase,
    "lowercase_latin": lowercase_latin,
    "lowercase_polish": lowercase_polish,
    "lowercase": lowercase,
    "letters": letters,
    "latin_letters": latin_letters,
    "polish_letters": polish_letters,
    "special_characters": special_characters,
    "whitespaces": whitespaces,
    "non_letters": non_letters,
    "full_dataset": full_dataset } 

#get --dataset or -d argument
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="dataset to use",
    choices=dataset_options.keys(), default="full_dataset")
args = parser.parse_args()

dataset = list(dataset_options[args.dataset])

