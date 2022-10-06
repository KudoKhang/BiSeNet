import sys
import time

a = 0
for x in range (0,12):
    a = a + 1
    b = ("Loading" + "." * a)
    # \r prints a carriage return first, so `b` is printed on top of the previous line.
    logs = f"{'Detect person':<20} {x:<20}"
    sys.stdout.write('\r' + logs)
    time.sleep(0.1)