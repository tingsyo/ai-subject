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
from google import genai
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

#from utils import config

#-----------------------------------------------------------------------
def respond_likert_item_gemini(item, instruction, persona="", random_seed=None):
    ''' Use google-genai API to generate a response to a Likert-scale item. '''
    # Customized config
    cfg = {
            "temperature":0.0,
            "responseSchema":{
                "type":"object",
                "properties":{
                    "rating":{"type":"string"},
                    "reasoning":{"type":"string"}
                },
                "required":["rating","reasoning"]
            },
        }        
    if not random_seed is None:
        cfg["seed"] = random_seed
    # Format instruction
    format_instruction = "\n Output the results as a json object:\
    {'rating':'1-5' ,'reasoning':'a simple reasoning of the rating in one sentence'}"
    # Call API
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents=persona+instruction+item+format_instruction,
        config=cfg
    )
    return(json.loads(response.text))

def respond_likert_questionnaire_gemini(questionnaire, instruction, persona="", random_seed=None):
    ''' Use google-genai API to generate a responses to a Likert-scale questionnaire. 
          - questionnaire: a list of strings that contains the questionnaire.
          - persona: a string that contains the persona of the respondent.
          - instruction: a string that contains the instructions for ansowering the questionnaire.
          - random_seed: int, rnadom seed to control the generation.
          - return: a list ofdictionary that contains the generated responses to the questionnaire.
    '''
    results = []
    for item in questionnaire:
        results.append(respond_likert_item_gemini(item, instruction, persona, random_seed))
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
        logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)
    # Load configuration file
    with open(args.config, 'rb') as f:
        logging.info('Loading configuration.')
        config = tomllib.load(f)
        logging.info(config)
    # Read questionnaire file
    questionnaire = pd.read_csv(args.items)
    items = list(questionnaire['item'])
    # Generate responses
    results = respond_likert_questionnaire_gemini(items, config['instruction'], config['persona'], config['random_seed'])
    # Prepare output
    pd.DataFrame(results).to_csv(args.output)
    # done
    return(0)

#==========
# Script
#==========
if __name__ == "__main__":
    main()

