"""Initialize game and run main gameplay loop."""
import configparser
import random
import warnings

import rich
import wikipedia
from bs4 import GuessedAtParserWarning
from click import edit
from rich.prompt import IntPrompt, Prompt
from rich.traceback import install

from classification import LinearSVC
from game import Game
from utils import console

# initialize config and traceback module
config = configparser.ConfigParser()
config.read("./config.ini")
install()

# filter bs4's GuessedAtParserWarning and spacy's empty vector warning
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)
warnings.filterwarnings("ignore", message=r"\[W008\]")

# get list of "vital" articles
with open("articles.txt") as f:
    ARTICLES = [line.strip() for line in f.readlines()]


def main():
    """Run the main gameplay loop."""
    random_endpoints = start_menu()
    begin = "n"
    while begin != "y":
        start, end = init_endpoints(random_endpoints=random_endpoints)
        begin = Prompt.ask("Use these endpoints?", choices=["y", "n"], default="y")

    game = Game(start, end, config)

    while True:
        game.cmd = input(">>> ")
        if game.cmd != "":
            game.commands.get(game.cmd.split()[0], game.classify)()


def start_menu():
    """Show the start menu and handle user selection.

    Returns:
    boolean -- True if using random endpoints, False otherwise.
    """
    options = [
        "([green]1[/green]) New game with random pages",
        "([green]2[/green]) New game with defined pages",
        "([bright_yellow]3[/bright_yellow]) Train classifier",
        "([bright_yellow]4[/bright_yellow]) Settings",
        "([red]5[/red]) Exit",
    ]

    console.print(
        rich.panel.Panel(
            "\n".join(options),
            title="Welcome to Wikigame-NLP",
            border_style="blue",
            expand=False,
        )
    )

    tmp = 100
    while tmp > 2:
        tmp = IntPrompt.ask("Select from the options above")
        if tmp == 5:
            exit()
        elif tmp == 4:
            edit(filename="./config.ini")
            config.read("./config.ini")
        elif tmp == 3:
            LinearSVC.train(
                config["classifier"]["data_path"], config["classifier"]["model_path"]
            )

    return False if tmp == 2 else True


def init_endpoints(random_endpoints=True):
    """Define endpoints, and print them in a table format.

    Arguments:
    random_endpoints -- boolean, default True. Use random or manual endpoints.

    Returns:
    Two wikipedia `Page` objects.
    """

    # get two random vital articles...
    if random_endpoints:
        start = get_random_vital()
        end = get_random_vital()
    # ...or prompt the user to input two titles, and get corresponding pages
    else:
        start = wikipedia.page(
            Prompt.ask("Page to use as the [blue]start[/blue] point")
        )
        end = wikipedia.page(Prompt.ask("Page to use as the [blue]end[/blue] point"))

    # get page summaries
    start_sum = start.summary
    end_sum = end.summary

    # if summary is longer than 280 characters, truncate and add ellipsis
    if len(start_sum) > 280:
        start_sum = start_sum[:280] + "[blue][...]"
    if len(end_sum) > 280:
        end_sum = end_sum[:280] + "[blue][...]"

    # present the endpoints with rich.Table
    table = rich.table.Table()
    table.add_column(f"Start: [blue]{start.title}", ratio=0.5)
    table.add_column(f"End: [blue]{end.title}", ratio=0.5)
    table.add_row(start_sum, end_sum)
    console.print(table)

    return start, end


def get_random_vital():
    """Get a random vital article."""
    title = random.choice(ARTICLES)
    return wikipedia.page(title, auto_suggest=False)


if __name__ == "__main__":
    main()
