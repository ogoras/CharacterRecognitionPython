# -*- coding: utf-8 -*-
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
        return '\r\n'

def classno_to_charname(ind):
    char = classno_to_char(ind)
    if char == ' ':
        return 'space'
    elif char == '\t':
        return 'tab'
    elif char == '\r\n':
        return 'newline'
    else:
        return char
