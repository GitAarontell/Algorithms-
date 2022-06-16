import unittest
from main import product_sum
from main import knapsack
from main import branchSums
from main import BST
from main import spiralTraverse


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

    def test_branch_sum(self):
        tree = BST(1)
        tree.left = BST(2)
        tree.right = BST(3)
        expected_output = [3, 4]
        self.assertEqual(branchSums(tree), expected_output)

    def test_spiral_traverse_01(self):
        a = [
            [1, 2, 3, 4],
            [12, 13, 14, 5],
            [11, 16, 15, 6],
            [10, 9, 8, 7]
        ]
        expected_output = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.assertEqual(spiralTraverse(a), expected_output)

    def test_spiral_traverse_02(self):
        a = [
            [4, 2, 3, 6, 7, 8, 1, 9, 5, 10],
            [12, 19, 15, 16, 20, 18, 13, 17, 11, 14]
        ]
        expected_output = [4, 2, 3, 6, 7, 8, 1, 9, 5, 10, 14, 11, 17, 13, 18, 20, 16, 15, 19, 12]
        self.assertEqual(spiralTraverse(a), expected_output)


if __name__ == '__main__':
    unittest.main()
