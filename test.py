import unittest
from main import product_sum


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


if __name__ == '__main__':
    unittest.main()