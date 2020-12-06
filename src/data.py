import time
import string
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter

import requests
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm 
from scholarly import scholarly, ProxyGenerator
from statsmodels.distributions.empirical_distribution import ECDF

from src.latex_accents import AccentConverter
from src.utils import (COMMUNITY2CONF, CONF2COMMUNITY, YEAR_RANGE, INTERESTS,
                       FUTURE_YEAR,
                       clean_author_names, strip_accents,
                       standaraize_author_name, is_sorted, unify_author)



DBLP_MAX_HITS = 1000
DBPL_CONF_JSON_URL = f'https://dblp.org/search/publ/api?q=toc%3Adb/conf/{{name}}/{{name}}{{year}}.bht%3A&h={DBLP_MAX_HITS}&f={{f}}&format=json'

SCHOLARLY_AUTHOR_FIELDS = ['affiliation', 'citedby', 'citedby5y', 'cites_per_year', 'email',
                 'hindex', 'hindex5y','i10index', 'i10index5y', 'id', 'interests', 'url_picture']


logger = logging.getLogger(__name__)


def scrap_dblp_conf_json(name, year):

    papers = []
    next_first = 0
    data = []
    
    while not next_first or next_first == DBLP_MAX_HITS:
        url = DBPL_CONF_JSON_URL.format(name=name, year=year, f=next_first)
        r = requests.get(url)
        data = json.loads(r.text)
        
        hits = data['result']['hits']['hit']
        for hit in hits:
            papers.append(hit['info'])

        next_first = len(hits)
    
    return papers


def extract_authors(authors_dblp):
    if isinstance(authors_dblp, dict):
        return [authors_dblp['text']]
    else:
        return [author['text'] for author in authors_dblp]


def retrive_conf_papers(name, year):
    
    papers = scrap_dblp_conf_json(name, year)

    df = pd.DataFrame(papers)
    df = df[df['type'] == 'Conference and Workshop Papers']
    df = df.drop('type', axis=1)

    n_missing_author = df['authors'].isna().sum()
    if n_missing_author:
        logger.warning(f'Missing authors ({name}, {year}): {n_missing_author}')
    df = df.dropna(subset=['authors'])

    df['authors_dblp'] = df['authors'][:]
    df['authors'] = df['authors'].apply(lambda r: r['author']).apply(extract_authors)
    df['authors'] = df['authors'].apply(lambda r: [clean_author_names(author) for author in r])

    df['stand_authors'] = df['authors'].apply(lambda authors:
                                                   [standaraize_author_name(author)
                                                    for author in authors])

    df['is_alphabetical'] = df['stand_authors'].apply(lambda authors: is_sorted(authors))
    df['n_author'] = df['authors'].apply(lambda r: len(r))

    df['conf'] = name.upper()
    df['community'] = df['conf'].apply(lambda r: CONF2COMMUNITY[r])

    return df


def build_papers_df(conf2community, year_range):
    papers_df = pd.concat([retrive_conf_papers(conf.lower(), year)
                           for conf in tqdm(conf2community)
                           for year in year_range],
                         ignore_index=True)
    
    return papers_df


def make_paper_dataset(path=None, conf2community=CONF2COMMUNITY, year_range=YEAR_RANGE):
    json_path = None
    if path is not None:
        json_path = Path(path) / 'papers_{}_{}_{}.json'.format('_'.join(sorted(conf2community)),
                                                         year_range[0], year_range[-1])
        if json_path.exists():
            return pd.read_json(json_path)
        
    papers_df = build_papers_df(conf2community, year_range)
    
    if json_path is not None:
        papers_df.to_json(json_path)
    
    return papers_df



def community_ratio(row):
    if row['community'] == 'ML':
        return np.inf
    elif row['community'] == 'TCS':
        return 0
    else:
        count = Counter(row['communities'])
        return count['ML'] / count['TCS']


def identiy_major_community(community_ratio, threshold=1.5):
    if community_ratio >= threshold:
        return 'ML'
    elif community_ratio <= 1/threshold:
        return 'TCS'
    else:
        return 'BOTH'

    
def get_coauthors_letters(row):
    return [a[0][0]
            for ca in row['stand_coauthors']
            for a in ca if a != row['stand_name']]  # do not need that


def build_authors_df(papers_df):

    authors_d = defaultdict(lambda: defaultdict(list))

    for _, paper in papers_df.iterrows():
        for author in paper['authors']:
            authors_d[author]['stand_name'] = standaraize_author_name(author)
            authors_d[author]['confs'].append(paper['conf'])
            authors_d[author]['communities'].append(paper['community'])

    authors_df = pd.DataFrame.from_dict(authors_d, orient='index')
    authors_df['community'] = authors_df['communities'].apply(set).apply(lambda c: 'BOTH' if len(c) > 1 else list(c)[0])

    authors_df['community_ratio'] = authors_df.apply(lambda r: community_ratio(r), axis=1)
    authors_df['major_community'] = authors_df['community_ratio'].apply(lambda r: identiy_major_community(r))
    
    authors_df['multi'] = authors_df.index.map(lambda name: '0' in name)

    authors_df['real_name'] = authors_df.index.map(lambda name: name
                                                     if '0' not in name
                                                     else name.rsplit(maxsplit=1)[0])

    canonical_names = defaultdict(list)

    for author, row in authors_df.iterrows():
        if '0' in author:
            real_name = row['real_name']
            canonical_names[real_name].append(author)

    count_multi_real_name = Counter(map(len, canonical_names.values()))
    logging.warning(f'Multiple author name to drop: {count_multi_real_name}')
    authors_to_drop = set.union(*[set(name_number_s)
                                  for name_number_s in canonical_names.values()
                                  if len(name_number_s) > 1])
    authors_df = authors_df.drop(authors_to_drop)


    ########################################################
    
    
    coauthors = defaultdict(lambda: defaultdict(list))

    for _, paper in papers_df.iterrows():
        for name in paper['authors']:
            coauthors[name]['papers_authors'].append([author for author in paper['authors']])
            coauthors[name]['coauthors'].append([author for author in paper['authors'] if name != author])


    authors_df = pd.merge(authors_df,
                          pd.DataFrame.from_dict(coauthors, orient='index'),
                          how='left', left_index=True, right_index=True)

    authors_df['n_pub'] = authors_df['papers_authors'].apply(len)
    authors_df['n_coauthor'] = authors_df['coauthors'].apply(lambda r: len(set.union(*map(set, r))))

    authors_df['stand_coauthors'] = authors_df['coauthors'].apply(lambda r: [[standaraize_author_name(a) for a in ca] for ca in r])

    authors_df['avg_is_sort'] = authors_df['stand_coauthors'].apply(lambda r: np.mean([is_sorted(ca) for ca in  r]))

    authors_df['only_sing'] = authors_df['n_coauthor'].apply(lambda r: r == 0)

    authors_df['coauthors_letter'] = authors_df.apply(get_coauthors_letters, axis=1)

    authors_df['positions'] = authors_df.apply(lambda r: [(pa.index(r.name), len(pa))
                                                      for pa in r['papers_authors']
                                                      if len(pa) > 1],
                          axis=1)

    (authors_df['first_prop'],
     authors_df['last_prop']) = (authors_df['positions'].apply(lambda pos: np.mean([ind == 0 for ind, _ in pos])),
                                authors_df['positions'].apply(lambda pos: np.mean([ind == (n_au-1) for ind, n_au in pos])))
    
    ########################################################
   
    authors_df['letter'] = authors_df['stand_name'].str[0]
    authors_df['letter_int'] = authors_df['letter'].apply(string.ascii_uppercase.index)
    letter_ecdf = ECDF(authors_df['letter_int'])
    authors_df['letter_val'] = authors_df['letter_int'].apply(letter_ecdf)


    return authors_df


def scrape_google_scholar(authors_df, sleep=1/3, interests=INTERESTS, with_tor_proxy=False):
    if with_tor_proxy:
        pg = ProxyGenerator()
        pg.Tor_Internal(tor_cmd='tor')
        scholarly.use_proxy(pg)
        print('Using Tor!')

    results = {}
    
    try:
        author = None  # UGLY HACK!!!

        for name, row in tqdm(authors_df.iterrows(), total=len(authors_df)):
            search_query = scholarly.search_author(row['real_name'])
            at_least_one = False
            for author in search_query:
                at_least_one = True
                if not row['multi']:
                    break
                else:
                    author_interests = ' '.join(author.interests).lower()
                    if (any(interest in author_interests for interest in interests)
                        or 'comput' in author.affiliation.lower()):
                        break
                    else:
                        results[name] = {'status': 'fail_multi'}
                time.sleep(sleep)
            else:
                results[name] = {'status': 'fail_not_found'}

            if at_least_one:
                results[name] = dict(status='success', obj=author.fill(['indices', 'counts']))

            time.sleep(sleep)
    except Exception as e:
        print(e)
            
    finally:    
        return results


def flatten_scholarly_result(info):
    d = {'status': info['status']}
    if info['status'] == 'success':
        d.update({field: getattr(info['obj'], field)
                            for field in SCHOLARLY_AUTHOR_FIELDS if hasattr(info['obj'], field)})
    return d


def enrich_author_df_with_scholarly(results, authors_df):

    authors_info_d = {name: flatten_scholarly_result(info)
               for name, info in results.items()}

    authors_info_df = pd.DataFrame.from_dict(authors_info_d, orient='index')
    success_mask = (authors_info_df['status'] == 'success') 
    authors_info_df.loc[success_mask, ['citedby', 'citedby5y']] = authors_info_df.loc[success_mask, ['citedby', 'citedby5y']].fillna(0)
    authors_info_df.loc[success_mask, 'scientific_age'] = (authors_info_df.loc[success_mask, 'cites_per_year']
                                                         .apply(lambda r: FUTURE_YEAR - min(r.keys()) if r else 0) + 1)

    authors_info_df.loc[success_mask, 'annual_productivity'] = (authors_info_df.loc[success_mask, 'citedby']
                                                               / authors_info_df.loc[success_mask, 'scientific_age'])
    authors_info_df.loc[success_mask, 'annual_productivity'] = (authors_info_df.loc[success_mask, 'annual_productivity']
                                                               .fillna(0))
    assert set(authors_info_df.index).issubset(set(authors_df.index))
    
    return pd.merge(authors_df, authors_info_df, how='left', left_index=True, right_index=True)
    