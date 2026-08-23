import re
from dataclasses import dataclass


@dataclass
class TNVEDNode:
    code: str
    description: str
    level: int


class TNVEDKnowledge:
    """
    Парсер tnved_knowledge.txt
    """

    CODE_RE = re.compile(
        r"^\s*(\d{2,10})\s*[|\-]"
    )

    LEVEL_MAP = {
        2: 1,
        4: 2,
        6: 3,
        10: 4
    }


    def __init__(self, filepath):

        self.filepath = filepath

        self.nodes = self._parse()

        self.by_code = {
            node.code: node
            for node in self.nodes
        }


    def _parse(self):

        nodes = []

        with open(
            self.filepath,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue


                match = self.CODE_RE.match(line)

                if not match:
                    continue


                code = match.group(1)

                description = (
                    line[match.end():]
                    .strip()
                )

                description = re.sub(
                    r"^[\s\-–]+",
                    "",
                    description
                )


                if len(code) in (2, 4, 6):

                    nodes.append(
                        TNVEDNode(
                            code=code,
                            description=description,
                            level=self.LEVEL_MAP.get(
                                len(code),
                                0
                            )
                        )
                    )

        return nodes



    def get_context(self, code):

        code = str(code)


        result = []


        for length in [2,4,6]:

            prefix = code[:length]


            node = self.by_code.get(prefix)


            if node:

                result.append(
                    f"{node.code} {node.description}"
                )


        return " | ".join(result)