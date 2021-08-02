import signal, getch


def interrupted(signum, frame):
    return


def input():
    try:
        foo = getch.getch()
        return foo
    except:
        return


def waitKey(timeout):
    """
    the same function as cv2.waitKey()
    :param timeout:
    :return:
    """
    # set alarmd
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, timeout / 1000)
    s = input()
    # disable the alarm after success
    signal.setitimer(signal.ITIMER_REAL, 0)

    return s
