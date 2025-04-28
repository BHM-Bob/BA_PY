'''
Date: 2024-04-24 11:11:58
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-04-01 20:01:31
Description: 
'''
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

        pool = TaskPool().start()
        pool.add_task("task1", example_coroutine, "task1", 1)
        pool.add_task("task2", example_coroutine, "task2", 2)
        
        self.assertEqual(pool.count_done_tasks(), 0)  # Output: 0
        self.assertEqual(pool.query_task("task1"), TaskStatus.NOT_FINISHED)  # Output: TaskStatus.NOT_FINISHED
        self.assertEqual(pool.query_task("task2"), TaskStatus.NOT_FINISHED)  # Output: TaskStatus.NOT_FINISHED

        # wait for tasks to finish
        time.sleep(10)
        
        self.assertEqual(pool.count_done_tasks(), 2)  # Output: 2
        self.assertEqual(pool.query_task("task1"), 'Coroutine task1 result')  # Output: Coroutine task1 finished after 3 seconds
        self.assertEqual(pool.query_task("task2"), 'Coroutine task2 result')  # Output: Coroutine task2 finished after 5 seconds
        
    # def test_err(self):
    #     async def example_coroutine(name, seconds):
    #         print(f"Coroutine {name} start to raise error")
    #         raise ValueError("Error occurred in coroutine")
            
    #     with self.assertRaises(ValueError):
    #         pool = TaskPool().run()
    #         pool.add_task("task1", example_coroutine, "task1", 2)
    #         pool.add_task("task2", example_coroutine, "task2", 4)

    #         self.assertEqual(pool.query_task("task1"), TaskStatus.NOT_FINISHED)  # Output: TaskStatus.NOT_FINISHED
    #         self.assertEqual(pool.query_task("task2"), TaskStatus.NOT_FINISHED)  # Output: TaskStatus.NOT_FINISHED

    #         # wait for tasks to finish
    #         time.sleep(10)

    #         self.assertEqual(pool.query_task("task1"), TaskStatus.NOT_SUCCEEDED)  # Output: TaskStatus.NOT_SUCCEEDED
    #         self.assertEqual(pool.query_task("task2"), TaskStatus.NOT_SUCCEEDED)  # Output: TaskStatus.NOT_SUCCEEDED
        
        
class TaskPoolThreadTest(unittest.TestCase):
    def test_1(self):
        def example_function(name, seconds):
            print(f"Thread {name} started")
            time.sleep(seconds)
            print(f"Thread {name} finished after {seconds} seconds")
            return f'Thread {name} result'
        
        pool = TaskPool('thread').start()
        pool.add_task("task1", example_function, "task1", 1)
        pool.add_task("task2", example_function, "task2", 2)

        self.assertEqual(pool.query_task("task1"), TaskStatus.NOT_FINISHED)  # Output: TaskStatus.NOT_FINISHED
        self.assertEqual(pool.query_task("task2"), TaskStatus.NOT_FINISHED)  # Output: TaskStatus.NOT_FINISHED

        # wait for tasks to finish, thread mode must wait long
        time.sleep(6)

        self.assertEqual(pool.query_task("task1"), 'Thread task1 result')  # Output: task1 finished after 3 seconds
        self.assertEqual(pool.query_task("task2"), 'Thread task2 result')  # Output: task2 finished after 5 seconds
        
    def test_non_name(self):
        def example_function(name, seconds):
            print(f"Thread {name} started")
            time.sleep(seconds)
            print(f"Thread {name} finished after {seconds} seconds")
            return f'Thread {name} result'
        
        pool = TaskPool('thread').start()
        pool.add_task('', example_function, "task1", 1)
        pool.add_task('', example_function, "task2", 2)
        
        names = list(pool.tasks.keys())
        self.assertEqual(pool.query_task(names[0]), TaskStatus.NOT_FINISHED)  # Output: TaskStatus.NOT_FINISHED
        self.assertEqual(pool.query_task(names[1]), TaskStatus.NOT_FINISHED)  # Output: TaskStatus.NOT_FINISHED
        
        time.sleep(6)
        
        self.assertEqual(pool.query_task(names[0]), f'Thread task1 result')  # Output: task1 finished after 3 seconds
        self.assertEqual(pool.query_task(names[1]), f'Thread task2 result')  # Output: task2 finished after 5 seconds
        
        
class TaskPoolThreadsTest(unittest.TestCase):
    def test_n_worker_4(self):
        
        def example_function(name, seconds):
            print(f"Threads {name} started")
            time.sleep(seconds)
            print(f"Threads {name} finished after {seconds} seconds")
            return f'Threads {name} result'
        
        n_woker = 4
        pool = TaskPool('threads', n_woker).start()
        for i in range(n_woker):
            pool.add_task(f"task{i+1}", example_function, f"task{i+1}", 1)

        for i in range(n_woker):
            self.assertEqual(pool.query_task(f"task{i+1}"), TaskStatus.NOT_FINISHED)

        # wait for tasks to finish
        time.sleep(6)
        
        for i in range(n_woker):
            self.assertEqual(pool.query_task(f"task{i+1}"), f'Threads task{i+1} result')
        
        pool.close(1)
        
        
def TaskPoolProcessTest_example_function(name, seconds):
    print(f"Process {name} started")
    time.sleep(seconds)
    print(f"Process {name} finished after {seconds} seconds")
    return f'Process {name} result'

class TaskPoolProcessTest(unittest.TestCase):
    def test_n_worker_4(self):
        n_worker = 4
        pool = TaskPool('process', n_worker).start()
        for i in range(n_worker):
            pool.add_task(f"task{i+1}", TaskPoolProcessTest_example_function, f"task{i+1}", 2)

        for i in range(n_worker):
            self.assertEqual(pool.query_task(f"task{i+1}"), TaskStatus.NOT_FINISHED)

        # wait for tasks to finish
        results = pool.wait_till_tasks_done([f"task{i+1}" for i in range(n_worker)])
        
        for i in range(n_worker):
            self.assertEqual(results[f"task{i+1}"], f'Process task{i+1} result')
        
        pool.close(1)
        
        
class TaskPoolMapTest(unittest.TestCase):
    def test_list(self):
        def example_function(x, y):
            return x+y
        
        pool = TaskPool('threads', 4).start()
        results = pool.map_tasks(list(zip([[1]]*9, [{}]*9)), example_function, y=1)
        self.assertEqual(results, [2]*9)
        
        pool.close(1)
        
    def test_dict(self):
        def example_function(x, y):
            return x+y
        
        pool = TaskPool('threads', 4).start()
        results = pool.map_tasks({'a': ([1], {}), 'b': ([2], {}), 'c': ([3], {})}, example_function, y=1)
        self.assertEqual(results, {'a': 2, 'b': 3, 'c': 4})
        
        pool.close(1)
        
    def test_list_batch(self):
        def example_function(x, y):
            print(x, y)
            return sum(list(map(lambda i: i[0][0], x)))+y
        
        pool = TaskPool('threads', 4).start()
        results = pool.map_tasks(list(zip([[1]]*9, [{}]*9)), example_function, 3, y=1)
        self.assertEqual(results, [4]*3)
        
        pool.close(1)
        
    def test_list_batch2(self):
        def example_function(x, y):
            print(x, y)
            return sum(x)+y
        
        pool = TaskPool('threads', 4).start()
        results = pool.map_tasks(list(range(9)), example_function, 3, y=1)
        self.assertEqual(results, [sum(i)+1 for i in split_list(list(range(9)), 3)])
        
        pool.close(1)


if __name__ == '__main__':
    unittest.main()