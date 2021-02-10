# Wikigame-NLP

Wikigame-NLP is a showcase of various ML/NLP techniques centered on the "Wikipedia game". In this game, the player tries to go from one Wikipedia article to another solely by following links. At each step, the player can execute a number of commands; including basic ones such as visiting another page, and ML-related ones such as highlighting named entities.

A screenshot preview of a short game can be found in the `imgs` folder.

Wikigame-NLP is in its early stages and is still a work in progress.

## Getting Started

Create a Python virtual environment and activate it:  
`python3 -m venv env-name`  
`source env-name/bin/activate`

Clone this repository and install the required packages:  
`git clone https://github.com/NicTM/wikigame-nlp.git`  
`cd wikigame-nlp`  
`pip install -r requirements.txt`

Download the desired NLP model from spaCy. Make sure to set the correct model name in the configuration file `config.ini`. The default is "en_core_web_md".  
`python3 -m spacy download en_core_web_md`

The endpoints for the game are chosen from a list of "vital" articles. Build this list by running the following command:  
`python3 build_vital.py`

Run the game:  
`python3 wikigame.py`

If running for the first time, select option 3 ("train classifier") before starting the game.

## Commands

There are two ways of executing a command:  
    - Using the shorthand notation, in which case the command will be executed directly (e.g. `v Canada`)  
    - Using free-text input, in which case the program will try to classify the command appropriately (e.g. `follow the link to the article about Canada, please`). The classifier is a linear SVM.

Below is a list of available commands and their shorthand notation.

### visit - 'v'
Given the title of an article, visits the corresponding page. 

If using shorthand notation, anything after `v` is considered part of the title. If using free-text input, the program will try to detect the title with a simple 3-rule heuristic:  
    1. If the input contains quotation marks, use their contents as the title.  
    2. If the input contains one of the links in the current article, use it as the title.  
    3. If the input contains a term like "about" or "regarding", use the closest noun phrase to the right as the title.

If the page to be visited is not linked to the current article, the program will attempt to spellcheck it to the nearest valid link (using Jaro-Winkler similarity). The user can choose to go with the corrected title or the original, although the latter option "loses the game".

### back - 'b'
Goes back one or more pages.

If using shorthand notation, enter an integer after `b` to specify the number of pages (e.g. `b 2`). 

If using free-text input, the program will try to detect the number of pages by looking for cardinal numbers (e.g. `go back 2 pages` or `go back to page 2`) or ordinal numbers (e.g. `go back to the 2nd page`). Defaults to 1.

### history - 'hs'
Shows the history of visited pages.

### more - 'm'
Shows the entire contents of the current page.

### web - 'w'
Opens the current page in a web browser.

### links - 'l'
Shows the list of links contained in the current page. For a move to be valid, the player must follow one of these links.

### similar - 's'
Shows the list of links contained in the current page, sorted by semantic similarity to the end point's title.

Similarity is calculated using the spaCy model's word vectors. For multi-word titles, the average is taken.

### entities - 'e'
Highlights the named entities in the current page's summary. Entities are identified using the spaCy model.

### generate - 'g'
Passes the first X words of the summary as input to a generative model, which then attempts to "auto-complete" the text. Default X=25.

This function uses a HuggingFace pipeline with the "text-generation" task identifier. The model used by default is `distilgpt2`.

### quit - 'q'
Quits the game.

### help -'h'
Shows this document.
