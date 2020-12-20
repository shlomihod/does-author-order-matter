# Does Author Order Matter?
*by Shlomi Hod*

Final Course Project

BU - CS 591B2 - Fall 2020 - Networks and Markets: Theory and Applications

[Poster](https://docs.google.com/presentation/d/1zPEdF6HTi01h421h_IbihKtaBL_tXXDq5lP17i6AK_M/edit?usp=sharing)

## Abstract

There are two common norms of ordering author names in a scientific publication: Alphabetical order and contribution-based. Could the alphabetic order norm make the alphabetic rank of an author’s last name an irrelevant factor co-determining academic success? Previous work found such bias for authors in Economics - but does it hold in other disciplines? In the “Theory of Computer Science” (TCS) field, there is a strong alphabetic order norm: more than 97% of the papers in its two crown conferences, STOC and FOCS, between 2000 and 2009 follow it. Based on data collected from the DBLP and the Semantic Scholar, I test whether the last name’s alphabetic rank is correlated with academic success in TCS while controlling for “scientific age” and ethnicity. I find no such main effect, and I conclude that no evidence that the last name’s alphabetic rank is a factor in the academic success in the TCS community. Nevertheless, I identify possible limitations of my analysis and propose future directions to address them.


## Structure

- `data` - The data used for the research, exported from DBLP and Semantic Scholar
- `src` - Python files used to export the data
- `notebooks`
  - `data.ipynb` - Jupyter Notebook to reproduce the analysis from the paper.
  - `data.ipynb` - Jupyter Notebook to scrape the Semantic Scholar without passing the quota limit.
- `figures` - figures to the paper and poster.