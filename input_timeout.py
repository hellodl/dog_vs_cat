import signal
TIMEOUT = 5


class InputTimeoutError(Exception):
    pass


def interrupted(signum, frame):
    '''
    called when read time out
    :param signum: 
    :param frame: 
    :return: 
    '''
    raise InputTimeoutError


def input_to(str='', timeout = 0):
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(timeout)
    input_str = None
    try:
        input_str = input(str)
        signal.alarm(0)
    except InputTimeoutError:
        pass

    return input_str
