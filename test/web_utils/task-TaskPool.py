import asyncio
import time
import unittest

from mbapy.base import *
from mbapy.web_utils.task import *


class TaskPoolCorutineTest(unittest.TestCase):
    def test_1(self):
        async def example_coroutine(name, seconds):
            print(f"Coroutine {name} started")
            await asyncio.sleep(seconds)
            print(f"Coroutine {name} finished after {seconds} seconds")
            return f"Coroutine {name} result"

        pool = TaskPool().run()
        pool.add_task("task1", example_coroutine, "task1", 2)
        pool.add_task("task2", example_coroutine, "task2", 4)

        self.assertEqual(pool.query_task("task1"), TaskStatus.NOT_FINISHED)  # Output: TaskStatus.NOT_FINISHED
        self.assertEqual(pool.query_task("task2"), TaskStatus.NOT_FINISHED)  # Output: TaskStatus.NOT_FINISHED

        # wait for tasks to finish
        time.sleep(10)

        self.assertEqual(pool.query_task("task1"), 'Coroutine task1 result')  # Output: Coroutine task1 finished after 3 seconds
        self.assertEqual(pool.query_task("task2"), 'Coroutine task2 result')  # Output: Coroutine task2 finished after 5 seconds
        
        
class TaskPoolThreadTest(unittest.TestCase):
    def test_1(self):
        def example_function(name, seconds):
            print(f"{name} started")
            time.sleep(seconds)
            print(f"{name} finished after {seconds} seconds")
            return f'{name} result'
        
        pool = TaskPool('thread').run()
        pool.add_task("task1", example_function, "task1", 2)
        pool.add_task("task2", example_function, "task2", 4)

        self.assertEqual(pool.query_task("task1"), TaskStatus.NOT_FINISHED)  # Output: TaskStatus.NOT_FINISHED
        self.assertEqual(pool.query_task("task2"), TaskStatus.NOT_FINISHED)  # Output: TaskStatus.NOT_FINISHED

        # wait for tasks to finish, thread mode must wait long
        time.sleep(10)

        self.assertEqual(pool.query_task("task1"), 'task1 result')  # Output: task1 finished after 3 seconds
        self.assertEqual(pool.query_task("task2"), 'task2 result')  # Output: task2 finished after 5 seconds


if __name__ == '__main__':
    unittest.main()