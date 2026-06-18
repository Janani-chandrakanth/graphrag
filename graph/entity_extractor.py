import re

from graph.schemas import Relationship


def extract_relationships(text):

    relationships = []

    patterns = {
        "creates": "CREATES",
        "contains": "CONTAINS",
        "fulfills": "FULFILLS",
        "updates": "UPDATES",
        "generates": "GENERATES",
        "displays": "DISPLAYS"
    }

    lines = text.split("\n")

    for line in lines:

        line = line.strip()

        if not line:
            continue

        for keyword, relation_name in patterns.items():

            pattern = rf"(.+?) {keyword} (.+)"

            match = re.match(
                pattern,
                line,
                re.IGNORECASE
            )

            if match:

                relationships.append(
                    Relationship(
                        source=match.group(1).strip(),
                        relationship=relation_name,
                        target=match.group(2).strip().rstrip(".")
                    )
                )

    return relationships