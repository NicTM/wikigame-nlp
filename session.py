"""Class definition for `Game` object."""
import os
import re
import webbrowser

import joblib
import spacy
import wikipedia
from rich.panel import Panel
from rich.progress import track
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from transformers import logging, pipeline

import utils
from utils import console

# avoid/suppress HuggingFace warnings
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GameSession:
    def __init__(self, start, end, config):
        with console.status("Initializing..."):
            self.config = config

            # page variables -- current page, start point, end point, history
            self.page = start
            self.start = start
            self.end = end
            self.history = [(start.title, "bright_yellow")]

            # nlp variables -- spacy model, processed text, similarities
            self.nlp = spacy.load(config["spacy"]["model"])
            self.nlp_end = self.nlp(end.title)
            self.nlp_text = self.nlp(start.summary)
            self.sim = []

            # classifier
            self.clf = joblib.load(config["classifier"]["model_path"])

            # generator
            self.gen = None  # only loaded if necessary

            # miscellaneous variables
            self.commands = utils.commands(self)  # list of shortcuts/commands
            self.cmd = None  # latest command entered by user
            self.victory = False  # victory condition

        # print starting page
        self.new_page()

    def new_page(self):
        """Print a colored panel containing the page summary."""

        # get page title, with number (length of history)
        num_title = f"({len(self.history)}) {self.page.title}"

        # print panel with summary
        console.print("")
        console.print(
            Panel(
                self.page.summary.strip(),
                title=num_title,
                border_style=self.history[-1][1],
            )
        )

        # if user reached end point for the first time, print victory message
        if self.history[-1][1] == "green" and not self.victory:
            self.victory = True
            console.print(
                f":tada: You've reached the end point in {len(self.history)-1} moves! :tada:\n"
            )

        # reset similarities and process new summary
        self.sim = []
        self.nlp_text = self.nlp(self.page.summary)

    def visit(self):
        """Visit the specified page, with free-text title detection and spellcheck."""
        is_valid = True

        # get title from command
        title = utils.detect_title(self.cmd, self.page.links, self.nlp)

        # alert user and return if no title detected, otherwise print title
        if title is None:
            console.print("[red]Failed to detect the title of an article.[/red]")
            return
        else:
            console.print(f"Detected title: [blue]{title}[/blue]")

        # if title is not found in current page's links, attempt to correct title
        if title.lower() not in [link.lower() for link in self.page.links]:
            corrected = utils.correct_title(title, self.page.links)
            # if user cancels (`correct_title` returns None), return
            if corrected is None:
                return
            # if title was not corrected, flag page as not valid
            elif corrected == title:
                is_valid = False
            # if title was corrected, update variable
            else:
                title = corrected

        # apply appropriate color for the visited page
        if not is_valid or self.history[-1][1] == "red":
            color = "red"
        elif title.lower() == self.end.title.lower() or self.history[-1][1] == "green":
            color = "green"
        else:
            color = "bright_yellow"

        # try to visit new page
        page = utils.goto(title)
        # if successful, update history and print panel
        if page is not None:
            self.page = page
            self.history.append((self.page.title, color))
            self.new_page()

    def back(self):
        """Go back a certain number of pages, 1 by default."""

        # alert user if the current page is the start point
        if len(self.history) == 1:
            console.print("This is the starting page!")
            return

        # detect how many pages
        n = utils.detect_back_n(self.cmd, len(self.history))

        # modify current page and history, and print new page
        self.page = wikipedia.page(self.history[-(n + 1)][0], auto_suggest=False)
        self.history = self.history[:-n]
        self.new_page()

    def history_cmd(self):
        """Print a color-coded list of pages visited so far."""
        console.print(
            " > ".join(f"[{color}]{title}[/{color}]" for title, color in self.history)
        )

    def more(self):
        """Print the entire page content via pager."""
        with console.pager():
            console.print(self.page.content)

    def web(self):
        """Open the current page in the default web browser."""
        webbrowser.open(self.page.url, new=2)

    def links(self):
        """Print a list of links via pager."""
        with console.pager():
            console.print("\n".join(self.page.links))

    def similar(self):
        """Print a list of links via pager, sorted by semantic similarity to the end point."""

        # if the end point's title is out-of-vocabulary, alert user and return
        if self.nlp_end.vector_norm == 0:
            console.print(
                "Unable to compute similarities: end point is out-of-vocabulary (0 norm)."
            )
            return

        # compute similarities with spacy (unless they've already been computed)
        if self.sim == []:
            for link in track(self.page.links, description="Computing similarities..."):
                self.sim.append(self.nlp_end.similarity(self.nlp(link)))

        # print links and scores
        tmp = zip(self.sim, self.page.links)
        tmp = [f"{link} : {sim}" for sim, link in reversed(sorted(tmp))]
        with console.pager():
            console.print("\n".join(tmp))

    def entities(self):
        """Highlight named entities in the current page summary."""

        # alert user if no entities are found
        if self.nlp_text.ents == []:
            console.print("No entities found.")
            return

        # add blue color to tokens that are recognized as part of an entity
        text = Text()
        offset = 0
        for e in self.nlp_text.ents:
            text.append(self.page.summary[offset : e.start_char])
            text.append(self.page.summary[e.start_char : e.end_char], "bold blue")
            text.append(f" ({e.label_})", "blue")
            offset = e.end_char
        text.append(self.page.summary[offset:].strip())

        # print labelled summary
        title = f"({len(self.history)}) {self.page.title}"
        console.print(Panel(text, title=title, border_style=self.history[-1][1]))

    def generate(self):
        """Generate text following the first X (default X=25) words of the summary."""

        # if load_generator is false, alert user and return
        if not self.config.getboolean("generator", "load_generator"):
            console.print(
                "[red]Generation is unavailable as the model cannot be loaded.",
                "[red]To use this command, set 'load_generator' to true in the config file.",
            )
            return

        # otherwise, load generator model if it is still None
        elif self.gen is None:
            model_name = self.config["generator"]["model_name"]
            console.print(f"Loading model: [blue]{model_name}[/blue]")
            self.gen = pipeline("text-generation", model=model_name)

        # get variables from config
        len_input = self.config.getint("generator", "len_input")
        len_output = self.config.getint("generator", "len_output")
        num_outputs = self.config.getint("generator", "num_outputs")

        # get start of summary to use as generator input
        summary = " ".join(self.page.summary.split()[:len_input])

        with console.status("Generating..."):
            # pass input to model and capture generated text
            out_text = self.gen(
                summary,
                max_length=len_input + len_output,
                num_return_sequences=num_outputs,
            )
            out_text = [o["generated_text"] for o in out_text]
            # remove new lines repeated more than twice
            out_text = [re.sub("\n{3,}", "\n\n", o) for o in out_text]
            # try to avoid ending the generated text mid-sentence
            out_text = [o[: o.rfind(".") + 1] for o in out_text]

        # show generated text samples in a table format
        table = Table(show_lines=True, show_header=False)
        table.add_column()
        for o in out_text:
            table.add_row(summary + "[blue]" + o[len(summary) :])
        console.print(table)

    def quit(self):
        """Exit the session with user confirmation."""
        ask_quit = Prompt.ask(
            "Are you sure you want to quit?", choices=["y", "n"], default="n"
        )
        if ask_quit == "y":
            console.print("Bye! :wave:")
            exit()

    def help_cmd(self):
        """Show the help document (`README.md`)."""
        with open("./README.md") as f:
            with console.pager():
                console.print(f.read())

    def classify(self):
        """Classify free-text commands."""
        p = self.clf.predict([self.cmd])[0]
        c = max(self.clf.predict_proba([self.cmd])[0])
        console.print(
            f"Classifying command as [bold blue]{p}[/bold blue] with [bold blue]{round(c*100,2)}%[/bold blue] confidence."
        )
        self.commands[p]()
