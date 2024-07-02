import sys

stderr = sys.stderr

def syslog(fmt, *args, **kwargs):
    print(fmt.format(*args, **kwargs), file=stderr)
