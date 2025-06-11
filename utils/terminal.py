import shutil


def get_terminal_width():
    """Get the width of the terminal"""
    try:
        return shutil.get_terminal_size().columns
    except BaseException:
        return 80  # fallback to 80 if unable to detect


def print_header(text, char="="):
    """Print a centered header with full terminal width"""
    width = get_terminal_width()
    print(char * width)
    print(text.center(width))
    print(char * width)


def print_separator(char="-"):
    """Print a separator line with full terminal width"""
    width = get_terminal_width()
    print(char * width)
