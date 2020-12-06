import re
import string
import unicodedata


COMMUNITY2CONF = {'TCS': ['FOCS', 'STOC'],
                   'ML': ['NIPS', 'ICML']}

CONF2COMMUNITY = {conf: community
                  for community, confs in COMMUNITY2CONF.items()
                  for conf in confs}

YEAR_RANGE = list(range(2000, 2010))

PARENTHESES_RE = re.compile('\(\w+\)')

INTERESTS = ['coding', 'algorithm', 'combinatorics', 'program', 'software', 'verifi', 'automata',
             'database', 'data mining', 'optimization', 'network', 'distributed', 'security', 'crypto', 'cyber',
             'machine learning', 'comput', 'complexity', 'artificial', 'intelegence', 'deep learning']

FUTURE_YEAR = 2021

def clean_author_names(name):
    name = name.replace('&apos;', '\'')
    name = PARENTHESES_RE.sub('', name)
    name = ' '.join(name.split())
    return name


def strip_accents(text):
    return unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")


def standaraize_author_name(name):
    ascii_name = strip_accents(name).lower()
    ascii_name = ''.join(char.upper()
                         for char in name
                         if char in string.ascii_letters + ' ')
    return ' '.join(ascii_name.split()[::-1])


def is_sorted(seq):
    return all(seq[i] <= seq[i+1] for i in range(len(seq)-1))


def unify_author(authors):
    return set.union(*map(set, authors))

