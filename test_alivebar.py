from alive_progress import alive_bar
import time
a = 0
with alive_bar(theme='musical', length=200) as bar:
    for i in range(20):
        time.sleep(0.5)
        a += i
        bar()

