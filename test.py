import unittest
from main import product_sum
from main import knapsack


class TestCase(unittest.TestCase):
    # -------- Product Sum Algorithm testing -------- #

    def test_product_sum_01(self):
        numbers = [2, 3, 4]
        expected_output = 14
        self.assertEqual(product_sum(numbers), expected_output)

    def test_product_sum_02(self):
        numbers = [2, 2, 1, 3, 2, 1, 2, 2, 1, 2]
        expected_output = 19
        self.assertEqual(product_sum(numbers), expected_output)

    def test_product_sum_03(self):
        numbers = [2, 2, 1, 3, 2, 1, 2, 2, 1]
        expected_output = 17
        self.assertEqual(product_sum(numbers), expected_output)

    # -------- Knapsack Algorithm testing -------- #

    def test_knapsack_01(self):
        W = 5
        w = [1, 2, 3, 4]
        v = [10, 20, 5, 15]
        expected_output = 50
        self.assertEqual(knapsack(len(w), W, w, v), expected_output)

if __name__ == '__main__':
    unittest.main()