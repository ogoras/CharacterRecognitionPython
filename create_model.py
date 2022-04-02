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
parser.add_argument("--input", "-i", help="dataset folder",
    default="data/images")
args = parser.parse_args()

dataset = list(dataset_options[args.dataset])
dataset_folder = args.input

def classno_to_char(ind):
    polish_letters = list("ĄĆĘŁŃÓŚŹŻ")
    polish_miniscule_letters = list("ąćęłńóśźż")
    
    if ind < 10:
        return str(ind)
    elif ind < 36:
        return chr(ord('A') + ind - 10)
    elif ind < 45:
        return polish_letters[ind - 36]
    elif ind < 71:
        return chr(ord('a') + ind - 45)
    elif ind < 80:
        return polish_miniscule_letters[ind - 71]
    elif ind < 95:
        return chr(ord('!') + ind - 80)
    elif ind < 102:
        return chr(ord(':') + ind - 95)
    elif ind < 108:
        return chr(ord('[') + ind - 102)
    elif ind < 112:
        return chr(ord('{') + ind - 108)
    elif ind == 112:
        return ' '
    elif ind == 113:
        return '\t'
    else:
        return "\n"

def classno_to_charname(ind):
    char = classno_to_char(ind)
    if char == ' ':
        return 'space'
    elif char == '\t':
        return 'tab'
    elif char == '\n':
        return 'newline'
    else:
        return char

