#!/usr/bin/env python

import os


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__) + "/src/")

    from dinov2.train.train import main, get_args_parser
    args = get_args_parser(add_help=True).parse_args()
    main(args)
