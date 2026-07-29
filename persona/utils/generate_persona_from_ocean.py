#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script generates a text description of a person with 
the specified Big-Five Personality test scores.
'''
import argparse
import logging
import sys
import os
import json
import tomllib
from google import genai
import ollama
import pandas as pd

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2026~2028, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2026-07-15'

# Parameters used for generation
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
scores = [None, None, None, None, None]    # dummy data for INSTRUCTION_PROMPT
INSTRUCTION_PROMPT = """
Your task is to generate a highly realistic, nuanced personal profile for an individual based on their specific Big Five Personality Test scores (OCEAN).

### The Subject's Scores:
- **Openness to Experience:** {}% - (High: imaginative, curious, unconventional / Low: practical, traditional, routine-oriented)
- **Conscientiousness:** {}% - (High: organized, disciplined, goal-driven / Low: spontaneous, flexible, easily distracted)
- **Extraversion:** {}% - (High: outgoing, energetic, assertive / Low: reserved, reflective, socially selective)
- **Agreeableness:** {}% - (High: empathetic, trusting, cooperative / Low: competitive, skeptical, direct/blunt)
- **Neuroticism:** {}% - (High: sensitive, emotionally reactive, anxious / Low: resilient, calm, emotionally stable)

### Profile Requirements:
1. **Tone & Style:** Write the profile from a neutral, observant third-person perspective. Avoid clinical jargon; instead, focus on observable behaviors, internal motivations, and interpersonal dynamics.
2. **Nuance:** Do not just list the traits sequentially. Blend them together to show how they interact. (e.g., How does high Extraversion pair with low Agreeableness? It might create a competitive, highly vocal leader).
3. **Structure:** Divide the profile into the following four concise sections:
   - **The Snapshot:** A 2-3 sentence overview of who they are at a glance.
   - **Work & Productivity:** How they handle tasks, deadlines, stress, and professional collaboration.
   - **Social & Relationships:** How they interact with friends, partners, and strangers.
   - **Internal World:** Their typical thought patterns, coping mechanisms, and hidden struggles or strengths.

### Constraints:
- Do not explicitly name the percentages or use the words "Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism" in the final profile text. Show, don't tell.
- Avoid making the character a caricature; balance their strengths with realistic vulnerabilities.
"""

def generate_persona_gemini(scores, temperature=0.0, random_seed=None):
    ''' Use google-genai API to generate persona description. '''
    # Customized config
    cfg = {"temperature":temperature}        
    if not random_seed is None:
        cfg["seed"] = random_seed
    # Call API
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents=INSTRUCTION_PROMPT.format(scores[0],scores[1],scores[2],scores[3],scores[4]),
        config=cfg
    )
    output = {
        "model": "gemini-3.1-flash-lite",
        "temperature": temperature,
        "scores": scores,
        "persona": response.text,
    }
    return(output)

def generate_persona_ollama(scores, model='gemma4:latest',temperature=0.0):
    ''' Use google-genai API to generate persona description. '''
    # Call API
    response = ollama.generate(
        model=model,
        prompt=INSTRUCTION_PROMPT.format(scores[0],scores[1],scores[2],scores[3],scores[4]),
        options={"temperature":temperature}
    )
    output = {
        "model": model,
        "temperature": temperature,
        "scores": scores,
        "persona": response.response,
    }
    return(output)

def write_jsonl(data, filename):
    ''' Write a list of JSON object to JSONL file '''
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return(0)


#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Building convolutional autoencoder .')
    #parser.add_argument('--config', '-c', help='the configuration file in json format.')
    parser.add_argument('--input', '-i', default=None, help='the CSV file contains OCEAN scores in each row.')
    parser.add_argument('--model', '-m', default="gemma4:latest", help='the name of LLM.')
    parser.add_argument('--temperature', '-t', type=float, default=0.0, help='the temperature for LLM.')
    parser.add_argument('--output', '-o', help='the prefix of output files.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')

    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y%m%d %H:%M:%S',
            level=logging.INFO
    )
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    logging.info(args)
    # Read in OCEAN scores
    if not args.input is None:
        logging.info("Read OCEAN scores from file: "+args.input)
        scores = pd.read_csv(args.input)
        scores = scores.to_numpy().tolist()
    else:
    # Generate Scores
        logging.info("Create OCEAN scores with PR=[5, 35, 65, 95]")
        levels = [i for i in range(5,100,30)]
        scores = []
        for o in levels:
            for c in levels:
                for e in levels:
                    for a in levels:
                        for n in levels:
                            scores.append([o,c,e,a,n])
    # Loop through scores
    output = []
    counter = 0
    logging.info("Starting generation, total profile counts: "+str(len(scores)))
    for i in range(len(scores)):
        s = scores[i]
        logging.info("Count: "+str(i)+"\tScore: "+str(s))
        res = generate_persona_ollama(scores=s, model=args.model, temperature=args.temperature)
        output.append(res)
        # output every 100 responses
        if (i % 100 == 0) and (i!=0):
            fname = args.output + "_" + str(counter).zfill(4) + ".jsonl"
            logging.info("Output batch "+str(counter)+" to "+fname)
            write_jsonl(output, fname)
            counter = counter + 1
            output =[]
    # Output the final batch
    fname = args.output + "_" + str(counter).zfill(4) + ".jsonl"
    write_jsonl(output, fname)
    # done
    return(0)

#==========
# Script
#==========
if __name__ == "__main__":
    main()



