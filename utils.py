import re
import string

import numpy as np
import wikipedia
from jellyfish import jaro_winkler_similarity
from rich import print
from rich.columns import Columns
from rich.console import Console
from rich.prompt import IntPrompt, Prompt

console = Console()


def commands(game):
    """Given a Game object, return a dictionary of available commands."""
    return {
        "v": game.visit,
        "b": game.back,
        "hs": game.history_cmd,
        "m": game.more,
        "w": game.web,
        "l": game.links,
        "s": game.similar,
        "e": game.entities,
        "q": game.quit,
        "h": game.help_cmd,
    }


def detect_title(cmd, links, nlp):
    """Detect (and correct) the title when using the `visit` command.

    Arguments:
    cmd -- string; the latest user input.
    links -- list of strings; links in the current page.
    nlp -- spaCy model object.

    Returns:
    A string or None -- the detected title (if any).
    """

    # check if command starts with the shorthand `v`
    if cmd[:2] == "v ":
        return cmd[2:]

    # look for quotation marks, links, and manually-defined terms
    title = detect_title_quotes(cmd)
    if title is None:
        title = detect_title_links(cmd, links)
    if title is None:
        title = detect_title_about(cmd, nlp)

    return title


def detect_title_quotes(cmd):
    """Look for a title in quotation marks."""
    quot = re.compile("['\"“”‘’„”«»].+['\"“”‘’„”«»]")
    match = quot.search(cmd)
    if match is not None:
        return match.group()[1:-1]


def detect_title_links(cmd, links):
    """Look for a link contained within the command."""
    links.sort(key=len, reverse=True)  # look for longer links (subsequences) first
    for link in links:
        pattern = re.compile(r"(\W|^)" + re.escape(link.lower()) + r"(\W|$)")
        if pattern.search(cmd.lower()) is not None:
            return link


def detect_title_about(cmd, nlp):
    """Look for a noun phrase after a term like `about`."""

    # define list of terms that may precede a title
    ABOUT = [
        "about",
        "regarding",
        "concerning",
        "referring to",
        "relating to",
        "dealing with",
        "link",
        "subject",
        "topic",
        "matter",
        "page",
        "article",
        "summary",
    ]

    # in order, look for one of these terms in the command
    idx = -1
    for a in ABOUT:
        if f" {a} " in cmd:
            idx = cmd.index(f" {a} ")
            break
    if idx == -1:
        return None

    # get the closest noun phrase to the right, if any
    doc = nlp(cmd[idx + len(a) + 2 :])
    try:
        title = next(doc.noun_chunks)
    except StopIteration:
        return None

    # return the noun phrase as a title, without the leading determinant
    if title[0].pos_ == "DET":
        return title[1:].text
    else:
        return title.text


def correct_title(title, links):
    """Correct a title's spelling using Jaro-Winkler similarity, with confirmation.

    Arguments:
    title -- string; title to be corrected.
    links -- list of strings; candidate titles to compare to.

    Returns:
    A string or None -- the nearest valid link (None if cancelled).
    """

    # apply spellcheck method to title
    correction = spellcheck(title, links)
    # ask whether to use correction or original title (or cancel)
    options = {"y": correction, "n": title, "c": None}
    ask_correction = Prompt.ask(
        f"The page you're trying to visit is not linked to the current one. Did you mean [blue]{correction}[/blue]?",
        choices=["y", "n", "c"],
        default="c",
    )
    return options[ask_correction]


def spellcheck(title, links):
    """Calculate similarities between title and links; return the closest link."""
    distances = []
    for link in links:
        distances.append(jaro_winkler_similarity(title.lower(), link.lower()))
    return links[np.argmax(distances)]


def goto(title):
    """Given a title, visit the corresponding page with error handling."""

    # try to visit page
    try:
        page = wikipedia.page(title, auto_suggest=False)

    # in case of disambiguation, prompt user to select an option or cancel
    except wikipedia.exceptions.DisambiguationError as e:
        page = disambiguate(e)

    # in case of page error, alert user and return None
    except wikipedia.exceptions.PageError:
        console.print("[red]The page you're trying to visit does not exist.[/red]")
        page = None

    return page


def disambiguate(e):
    """Given a DisambiguationError, print the list of options and prompt for a selection."""

    # get options
    options = [
        opt
        for opt in e.options
        if not opt.startswith("All pages") and not opt.endswith("(disambiguation)")
    ]

    # assign numbers to options
    options_ = [f"({i + 1}) {opt}" for i, opt in enumerate(options)]
    # print options in column format
    print(Columns(options_, equal=True, expand=True))

    # prompt user for a selection
    print("You've reached a [cyan]disambiguation page[/cyan]!")
    tmp = IntPrompt.ask(
        "Enter a number to choose from the options above, or 0 to cancel",
        default=0,
    )

    # return None if canceled or invalid selection, otherwise return selected page
    if tmp < 1 or tmp > len(options) + 1:
        return None
    else:
        return wikipedia.page(options[tmp - 1], auto_suggest=False)


def detect_back_n(cmd, len_h):
    """Detect the number of pages when using the `back` command.

    Arguments:
    cmd -- string; the latest user input.
    len_h -- integer; length of the history.

    Returns:
    An integer -- the detected number of pages.
    """

    # remove punctuation, and split
    cmd = cmd.translate(str.maketrans("", "", string.punctuation))
    cmd = cmd.split()

    # return 1 if tmp is just one word (e.g. "back")
    if len(cmd) == 1:
        return 1

    n = 1
    for i, w in enumerate(cmd):
        # if one of the tokens is a digit `w`...
        if w.isdigit():
            # ...preceded by the word "page" or "article", n = (len_h - w)
            if i > 0 and cmd[i - 1] in ["page", "article"]:
                n = len_h - int(w)
            # ...otherwise, n = w
            else:
                n = int(w)
            break

    # n must be between 1 and (len_h - 1)
    n = max(1, n)
    n = min(n, len_h - 1)

    return n
