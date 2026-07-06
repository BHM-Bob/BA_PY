'''
Date: 2026-03-13
LastEditors: BHM-Bob
LastEditTime: 2026-03-13
Description: MultiQueue 可用性、正确性与性能测试
'''
import multiprocessing as mp
import random
import sys
import time
import unittest

from mbapy.web_utils.task import MultiQueue


def producer(q, count, pid):
    for i in range(count):
        q.put(f"p{pid}_i{i}")


def cleanup_processes():
    for p in mp.active_children():
        p.terminate()
        p.join(timeout=1)


class MultiQueueCorrectnessTest(unittest.TestCase):
    def test_basic_put_get(self):
        print("\n[test_basic_put_get] 测试基本 put/get...")
        mq = MultiQueue(n_queue=4)
        mq.put("item1")
        mq.put("item2")
        mq.put("item3")
        
        items = []
        while mq.qsize() > 0:
            item = mq.get()
            if item is not None:
                items.append(item)
        
        self.assertEqual(len(items), 3)
        self.assertIn("item1", items)
        self.assertIn("item2", items)
        self.assertIn("item3", items)
        print("[test_basic_put_get] 通过")
    
    def test_empty_and_full(self):
        print("\n[test_empty_and_full] 测试 empty...")
        mq = MultiQueue(n_queue=4)
        self.assertTrue(mq.empty())
        
        mq.put("item1")
        mq.put("item2")
        self.assertFalse(mq.empty())
        
        mq.get()
        mq.get()
        self.assertTrue(mq.empty())
        print("[test_empty_and_full] 通过")
    
    def test_qsize(self):
        print("\n[test_qsize] 测试 qsize...")
        mq = MultiQueue(n_queue=4)
        for i in range(10):
            mq.put(i)
        
        self.assertEqual(mq.qsize(), 10)
        
        for _ in range(5):
            mq.get()
        
        self.assertEqual(mq.qsize(), 5)
        print("[test_qsize] 通过")
    
    def test_random_isolation(self):
        print("\n[test_random_isolation] 测试随机数隔离...")
        random.seed(42)
        mq = MultiQueue(n_queue=4)
        
        mq.put("a")
        mq.put("b")
        mq.get()
        
        self.assertIsInstance(mq._rand, random.Random)
        
        random.seed(42)
        r1 = random.random()
        r2 = random.random()
        
        mq._rand.seed(42)
        mr1 = mq._rand.random()
        mr2 = mq._rand.random()
        
        self.assertEqual(r1, mr1)
        self.assertEqual(r2, mr2)
        print("[test_random_isolation] 通过")


class MultiQueuePerformanceTest(unittest.TestCase):
    def test_throughput_single_process(self):
        print("\n[test_throughput_single_process] 单进程吞吐量测试...")
        n_items = 10000
        n_queue = 8
        
        print(f"  测试 MultiQueue (n_queue={n_queue})...")
        mq = MultiQueue(n_queue=n_queue)
        start = time.time()
        for i in range(n_items):
            mq.put(i)
        put_time = time.time() - start
        print(f"    Put 完成: {put_time:.3f}s")
        
        start = time.time()
        for _ in range(n_items):
            mq.get()
        get_time = time.time() - start
        print(f"    Get 完成: {get_time:.3f}s")
        
        print(f"  MultiQueue Put: {n_items/put_time:.0f} items/s")
        print(f"  MultiQueue Get: {n_items/get_time:.0f} items/s")
        
        print(f"  测试标准 Queue...")
        std_mq = mp.Queue()
        start = time.time()
        for i in range(n_items):
            std_mq.put(i)
        put_time_std = time.time() - start
        print(f"    Put 完成: {put_time_std:.3f}s")
        
        start = time.time()
        for _ in range(n_items):
            std_mq.get()
        get_time_std = time.time() - start
        print(f"    Get 完成: {get_time_std:.3f}s")
        
        print(f"  标准 Queue Put: {n_items/put_time_std:.0f} items/s")
        print(f"  标准 Queue Get: {n_items/get_time_std:.0f} items/s")
    
    def test_throughput_multi_process_put(self):
        print("\n[test_throughput_multi_process_put] 多进程 Put 吞吐量测试...")
        n_items = 2000
        n_producers = 4
        n_queue = 8
        
        mq = MultiQueue(n_queue=n_queue)
        
        print(f"  MultiQueue: {n_producers} producers, {n_items} items each...")
        start = time.time()
        processes = []
        for i in range(n_producers):
            p = mp.Process(target=producer, args=(mq, n_items, i))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join(timeout=5)
        cleanup_processes()
        
        put_time = time.time() - start
        total_items = n_items * n_producers
        
        print(f"    完成: {put_time:.3f}s")
        print(f"  MultiQueue Put: {total_items/put_time:.0f} items/s, 队列大小: {mq.qsize()}")
        
        print(f"  标准 Queue: {n_producers} producers, {n_items} items each...")
        std_mq = mp.Queue()
        
        start = time.time()
        processes = []
        for i in range(n_producers):
            p = mp.Process(target=producer, args=(std_mq, n_items, i))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join(timeout=5)
        cleanup_processes()
        
        put_time_std = time.time() - start
        print(f"    完成: {put_time_std:.3f}s")
        print(f"  标准 Queue Put: {total_items/put_time_std:.0f} items/s")


if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    unittest.main(verbosity=2)
