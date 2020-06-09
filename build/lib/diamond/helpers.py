import os
import sys
import time
import datetime as dt


def progress(msg, level=0, last=None):
    """

    """

    # Determine indent
    indent = "    "*level
    t_curr = dt.datetime.now()

    # Determine diff
    if last == None:
        last = 0
        t_since = None
    else:
        t_since = t_curr - last

    # Make notif
    m = "{}{} :: {} :: since last : {}".format(
        indent,
        msg,
        t_curr,
        t_since
    )

    # Replace last with current
    last = t_curr
    

