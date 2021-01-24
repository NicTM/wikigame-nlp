"""Initialize list of vital articles (possible random endpoints)."""
import wikipedia
from rich.progress import track

topics = []

# Level 5 geography topics
topics.extend(
    [
        "5/Geography/" + x
        for x in [
            "Physical geography",
            "Countries and subdivisions",
            "Cities",
        ]
    ]
)

# Level 5 biology topics
topics.extend(
    [
        "5/Biological and health sciences/" + x
        for x in [
            "Biology, biochemistry, anatomy, and physiology",
            "Animals",
            "Plants, fungi, and other organisms",
            "Health, medicine, and disease",
        ]
    ]
)

# Level 5 physical science topics
topics.extend(
    [
        "5/Physical sciences/" + x
        for x in [
            "Basics and measurement",
            "Astronomy",
            "Chemistry",
            "Earth science",
            "Physics",
        ]
    ]
)

# Other categories; level 4 for people, level 5 otherwise
topics.extend(
    [
        "4/People",
        "5/History",
        "5/Arts",
        "5/Philosophy and religion",
        "5/Everyday life",
        "5/Everyday life/Sports, games and recreation",
        "5/Society and social sciences",
        "5/Technology",
        "5/Mathematics",
    ]
)


open("articles.txt", "w").close()
articles = open("articles.txt", "a")

for topic in track(topics, description="Building list of vital articles..."):
    links = wikipedia.page(f"Wikipedia:Vital articles/Level/{topic}").links
    # exclude timelines and disambiguations
    links = [
        link
        for link in links
        if "Timeline" not in topic and "disambiguation" not in topic
    ]
    articles.write("\n".join(links) + "\n")

articles.close()
