from unicodedata import east_asian_width
import re
import sys

try:
    _unicode = unicode
except NameError:
    _unicode = str

CUR_OS = sys.platform
IS_WIN = any(CUR_OS.startswith(i) for i in ['win32', 'cygwin'])
IS_NIX = any(CUR_OS.startswith(i) for i in ['aix', 'linux', 'darwin'])
RE_ANSI = re.compile(r"\x1b\[[;\d]*[A-Za-z]")


def _screen_shape_wrapper():  # pragma: no cover
    """
    Return a function which returns console dimensions (width, height).
    Supported: linux, osx, windows, cygwin.
    """
    _screen_shape = None
    if IS_WIN:
        _screen_shape = _screen_shape_windows
        if _screen_shape is None:
            _screen_shape = _screen_shape_tput
    if IS_NIX:
        _screen_shape = _screen_shape_linux
    return _screen_shape


def _screen_shape_windows(fp):  # pragma: no cover
    try:
        import struct
        from ctypes import create_string_buffer, windll
        from sys import stdin, stdout

        io_handle = -12  # assume stderr
        if fp == stdin:
            io_handle = -10
        elif fp == stdout:
            io_handle = -11

        h = windll.kernel32.GetStdHandle(io_handle)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
        if res:
            (_bufx, _bufy, _curx, _cury, _wattr, left, top, right, bottom,
             _maxx, _maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
            return right - left, bottom - top  # +1
    except Exception:  # nosec
        pass
    return None, None


def _screen_shape_tput(*_):  # pragma: no cover
    """cygwin xterm (windows)"""
    try:
        import shlex
        from subprocess import check_call  # nosec
        return [int(check_call(shlex.split('tput ' + i))) - 1
                for i in ('cols', 'lines')]
    except Exception:  # nosec
        pass
    return None, None


def _screen_shape_linux(fp):  # pragma: no cover

    try:
        from array import array
        from fcntl import ioctl
        from termios import TIOCGWINSZ
    except ImportError:
        return None
    else:
        try:
            rows, cols = array('h', ioctl(fp, TIOCGWINSZ, '\0' * 8))[:2]
            return cols, rows
        except Exception:
            try:
                return [int(os.environ[i]) - 1 for i in ("COLUMNS", "LINES")]
            except KeyError:
                return None, None


def len_str(text):
    text = RE_ANSI.sub('', text)
    len_str = sum(2 if east_asian_width(ch) in 'FW' else 1 for ch in _unicode(text))

    return len_str


def rstrip_until_endline(text, len, max_len, ):
    if len > max_len:
        over_len = len - max_len
        idx = -1
        while len_str(text[idx:]) < over_len:
            idx -= 1
        text = text[:idx]
    
    return text


def add_spaces_until_endline(text: str = "", shortening_end='..', align_right_side: bool = False):
    text = RE_ANSI.sub('', text)
    max_len = _screen_shape_wrapper()(sys.stdout)[0] + 1 - len_str(shortening_end)
    
    text2 = rstrip_until_endline(text, len_str(text), max_len)
    if shortening_end and text!=text2: 
        text2 = text2 + shortening_end

    del_len = sum(1 if east_asian_width(ch) in 'FW' else 0 for ch in _unicode(text2))

    return text2.ljust(max_len - del_len) if not align_right_side else text2.rjust(max_len - del_len)


def get_stdout_size():
    return _screen_shape_wrapper()(sys.stdout)


def is_notebook():
    try:
        from IPython import get_ipython
        return False if get_ipython() is None else True
    except:
        return False
