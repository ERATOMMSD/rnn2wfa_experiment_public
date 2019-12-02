import preserving_heapq
import unittest
import heapq


class TestPreservingHeapQueue(unittest.TestCase):
    def test_pop_push(self):
        q = preserving_heapq.PreservingHeapQueue(lambda p: p[0] ** 2 + p[1] ** 2)
        q.push((2, 3))
        q.push((-1, -1))
        q.push((1, 1))
        q.push((0, 0))
        self.assertListEqual(q.to_list(), [(0, 0), (-1, -1), (1, 1), (2, 3)])
        self.assertTupleEqual(q.pop()[0], (0, 0))
        self.assertTupleEqual(q.pop()[0], (-1, -1))
        self.assertTupleEqual(q.pop()[0], (1, 1))
        self.assertTupleEqual(q.pop()[0], (2, 3))
        self.assertEqual(q.pop(), None)

    def test_pop_push_with_prioirty(self):
        q = preserving_heapq.PreservingHeapQueue()
        f = lambda p: p[0] ** 2 + p[1] ** 2
        q.push_with_priority((2, 3), f((2, 3)))
        q.push_with_priority((-1, -1), f((-1, -1)))
        q.push_with_priority((1, 1), f((1, 1)))
        q.push_with_priority((0, 0), f((0, 0)))
        self.assertListEqual(q.to_list(), [(0, 0), (-1, -1), (1, 1), (2, 3)])

    def test_assertion_of_push(self):
        with self.assertRaises(preserving_heapq.NoPriorityFunctionExecption):
            q = preserving_heapq.PreservingHeapQueue()
            q.push((2, 3))

    def test_equiv_heapq(self):
        q = preserving_heapq.PreservingHeapQueue(lambda x: x)
        q2 = []
        q.push(2)
        heapq.heappush(q2, 2)
        q.push((-1))
        heapq.heappush(q2, -1)
        q.push(1)
        heapq.heappush(q2, 1)
        q.push(0)
        heapq.heappush(q2, 0)
        self.assertListEqual(q.to_list(), q2)

    def test_is_empty(self):
        q = preserving_heapq.PreservingHeapQueue(lambda x: x)
        self.assertTrue(q.is_empty())
        q.push(-1)
        self.assertFalse(q.is_empty())
        q.pop()
        self.assertTrue(q.is_empty())

    def test_pop_push_no_priority(self):
        q = preserving_heapq.PreservingHeapQueue(lambda p: 0)
        q.push((2, 3))
        q.push((-1, -1))
        q.push((1, 1))
        q.push((0, 0))
        q.push((0, 1))
        q.push((0, 2))
        self.assertTupleEqual(q.pop()[0], (2, 3))
        self.assertTupleEqual(q.pop()[0], (-1, -1))
        self.assertTupleEqual(q.pop()[0], (1, 1))
        self.assertTupleEqual(q.pop()[0], (0, 0))
        self.assertTupleEqual(q.pop()[0], (0, 1))
        self.assertTupleEqual(q.pop()[0], (0, 2))

        self.assertEqual(q.pop(), None)



if __name__ == "__main__":
    unittest.main()
