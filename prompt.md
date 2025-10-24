Can you please help me to put the encounter profile generation into a selfcontained python script?
I want you to use this script template:
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""filename.py

A consise

description

Author(s): author_name
"""

import argparse


#############
# Constants #
#############

CONSTANTS: str = "CONSTANT"
ARGS = None  # Global variable to store command line arguments


#########
# Input #
#########


def parse_args(args=None) -> argparse.Namespace:
    "Runtime args parser"
    parser = argparse.ArgumentParser("")

    parser.add_argument(
        "--argument",
        help="",
        type=str,
        required=True,
    )

    return parser.parse_args(args)


###########
# Helpers #
###########

#################
#     MAIN      #
#################


def main(args=None) -> None:
    # parse args
    global ARGS
    ARGS = parse_args(args)


if __name__ == "__main__":
    main()
```

Please make it self contained, so include also the code from utils.py

I want a nice an clean code that will in the end produce the encounter narrative just like the code in 04_notebook.ipynb does.
Everything that is not needed to produce the encounter narrative can be ommitted.
Please also adhere to the python style instructions.

