#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script read output of questionnaires and calculate the variability of responses.
'''
import argparse
import logging
import os
import pandas as pd

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2026~2028, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2026-06-25'


def read_responses(DATAPATH):
    files = os.listdir(DATAPATH)
    ## Separating rating and reasoning
    ratings = []
    reasonings = []
    # Read all files
    for f in files:
        tmp = pd.read_csv(DATAPATH+f, index_col=0)
        ratings.append(tmp['rating'])
        reasonings.append(tmp['reasoning'])
    # Create output
    df_ratings= pd.concat(ratings, axis=1).transpose()
    df_reasoning = pd.concat(reasonings, axis=1).transpose()
    return(df_ratings, df_reasoning)


def analyze_responses(rating, reasoning):
    std_response = rating.std()
    reasoning_0 = reasoning.iloc[0,:]
    reasoning_identical = (reasoning==reasoning_0).sum()
    results = pd.DataFrame({"std_rating":std_response,"identical_reasoning":reasoning_identical})
    return(results)

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Analyze the variability of the responses.')
    parser.add_argument('--input', '-i', help='the questionnaire result files in CSV format.')
    parser.add_argument('--output', '-o', help='the prefix of the output file.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')

    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)
    # Read questionnaire responses
    DATAPATH = args.input
    files = os.listdir(DATAPATH)
    logging.debug('Totally '+str(len(files))+' files.')
    logging.debug('Read the first file: '+DATAPATH+files[0])
    df = pd.read_csv(DATAPATH+files[0], index_col=0)
    logging.debug(df.shape)
    logging.debug(df.head())
    rating, reasoning = read_responses(DATAPATH)
    # Analyze responses
    results = analyze_responses(rating, reasoning)
    # Prepare output
    pd.DataFrame(results).to_csv(args.output, index=False)
    # done
    return(0)

#==========
# Script
#==========
if __name__ == "__main__":
    main()


