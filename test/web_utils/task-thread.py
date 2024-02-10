import time

from mbapy.web import *


class test_class:
    def test_func(self, a, b, c):
        print(a+b+c)
tc = test_class()
launch_sub_thread(statuesQue, [
    ('1', Key2Action('running', tc.test_func, [1, 1], {'c': 1})),
    ('1', Key2Action('running', tc.test_func, [1, 2], {'c': 1})),
    ('2', Key2Action('running', tc.test_func, [2, 2], {'c': 2})),
    ('3', Key2Action('running', tc.test_func, [3, 3], {'c': 3})),
])
while True:
    if statues_que_opts(statuesQue, '__is_quit__', 'getValue'):
        break
    time.sleep(1)