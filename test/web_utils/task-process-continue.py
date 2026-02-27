'''
Date: 2025-05-27 16:20:46
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2026-02-26 19:27:07
Description: test whether the TaskPool can run continuely in sprocess mode
'''
import numpy as npest_fn, np.random.rand(10000))
        pool.wait_till(lambda: pool.count_waiting_tasks() == 0,
                       0.001,
from tqdm import tqdm
from mbapy.web_utils.task import TaskPool


def test_fn(data: np.ndarray):
    time.sleep(0.1)
    return data.mean()


if __name__ == '__main__':
    pool = TaskPool('process', 10).start()
    for i in tqdm(range(1000)):
        pool.add_task(i, test_fn, np.random.rand(10000))
        pool.wait_till(lambda: pool.count_waiting_tasks() == 0,
                       0.001, update_result_queue=False)
    time.sleep(1)
    for i in tqdm(range(1000)):
        test_fn(np.random.rand(10000))
        pool.add_task(-i-1, test_fn, np.random.rand(10000))
        pool.wait_till(lambda: pool.count_waiting_tasks() == 0,
                       0.001, update_result_queue=False)