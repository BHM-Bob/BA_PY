'''
Date: 2024-03-22 23:01:15
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-03-22 23:02:51
Description: 
'''
import asyncio

from mbapy.web_utils.task import CoroutinePool, TaskStatus


async def example_coroutine(name, seconds):
    print(f"Coroutine {name} started")
    await asyncio.sleep(seconds)
    print(f"Coroutine {name} finished after {seconds} seconds")
    return f"Coroutine {name} result"

pool = CoroutinePool().run()
pool.add_task("task1", example_coroutine, "task1", 3)
pool.add_task("task2", example_coroutine, "task2", 5)

print(pool.query_task("task1"))  # Output: TaskStatus.NOT_FINISHED
print(pool.query_task("task2"))  # Output: TaskStatus.NOT_FINISHED

# wait for tasks to finish
import time
time.sleep(6)

print(pool.query_task("task1"))  # Output: Coroutine task1 finished after 3 seconds
print(pool.query_task("task2"))  # Output: Coroutine task2 finished after 5 seconds