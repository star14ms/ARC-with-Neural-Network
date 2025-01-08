import os
from rich import get_console

from arc.utils.print import is_notebook


console = get_console()


def echo(*object, sep=" ", end=''):
    is_notebook_ = is_notebook()
    if is_notebook_:
        os.system(f'echo \"{sep.join(map(str, [*object]))}\"')
    else:
        with console.capture() as capture:
            console.print(*object, sep=sep, end=end)
        os.system(f'echo \"{capture.get()}\"')
