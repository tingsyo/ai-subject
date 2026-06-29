#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script creates LLM-based responses to the specified Likert-scale questionnaire.
The persona settings of the respondent, the model-name, temprature and the random seed
of the LLM, and questionnaire items can be specfied in the arguments.
'''
import argparse
import logging
import sys
import os
import json
import tomllib
import ollama
import pandas as pd

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2026~2028, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2026-06-15'

from pydantic import BaseModel, Field, ConfigDict
class LikertItem(BaseModel):
    rating: str = Field(description="Rating of the item.")
    reasoning: str = Field(description="Reasoning of the rating.")
#-----------------------------------------------------------------------
def respond_likert_item_ollama(item, instruction, persona="", random_seed=None):
    ''' Use Ollama API to generate a response to a Likert-scale item. '''
    format_instruction = "\n Output the results as a json object:\
    {'rating':'1-5' ,'reasoning':'a simple reasoning of the rating in one sentence'}"
    response = ollama.generate(
        model='gemma3:latest',
        prompt=persona+instruction+item+format_instruction,
        format=LikertItem.model_json_schema(),
        options={
            "seed":random_seed,
            "temperature":0.0,
        }
    )
    res = LikertItem.model_validate_json(response.response)
    return(res)

def respond_likert_questionnaire_ollama(questionnaire, instruction, persona="", random_seed=None):
    ''' Use Ollama API to generate a responses to a Likert-scale questionnaire. 
          - questionnaire: a list of strings that contains the questionnaire.
          - persona: a string that contains the persona of the respondent.
          - instruction: a string that contains the instructions for ansowering the questionnaire.
          - random_seed: int, rnadom seed to control the generation.
          - return: a list ofdictionary that contains the generated responses to the questionnaire.
    '''
    results = []
    for item in questionnaire:
        res = respond_likert_item_ollama(item, instruction, persona, random_seed)
        results.append({'rating':res.rating,'reasoning':res.reasoning})
    return(results)

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Building convolutional autoencoder .')
    parser.add_argument('--config', '-c', help='the configuration file in json format.')
    parser.add_argument('--items', '-i', help='the questionnaire file in json format.')
    parser.add_argument('--output', '-o', help='the prefix of output files.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')

    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.INFO)
    logging.debug(args)
    # Load configuration file
    with open(args.config, 'rb') as f:
        config = tomllib.load(f)
        logging.info('Loading configuration.')
    # Read questionnaire file
    questionnaire = pd.read_csv(args.items)
    items = list(questionnaire['item'])
    # Generate responses
    results = respond_likert_questionnaire_ollama(items, config['instruction'], config['persona'], config['random_seed'])
    # Prepare output
    pd.DataFrame(results).to_csv(args.output)
    # done
    return(0)

#==========
# Script
#==========
if __name__ == "__main__":
    main()

