import sys


# --------------- Fibonacci Algorithm --------------- #

# fibonacci top down
def fibonacci(n):
    cache = [-1] * (n + 1)
    cache[0], cache[1], cache[2] = 0, 1, 2
    return fibonacci_helper(n, cache)


def fibonacci_helper(n, cache):
    if cache[n] != -1:
        return cache[n]
    if n == 0 or n == 1:
        return 1
    else:
        cache[n] = fibonacci_helper(n - 1, cache) + fibonacci_helper(n - 2, cache)
        return cache[n]


def fibonacci_bottom_up_naive(n):
    if n == 0 or n == 1:
        return 1
    # a starts at the value of index 1 which is 1
    a = 1
    temp_val = 0
    # if n != 0 or 1, then set value to value at index 2
    value = 2
    for i in range(n - 2):
        temp_val = value
        value += a
        a = temp_val
    return value


def fibonacci_bottom_up(n):
    if n == 0 or n == 1:
        return 1

    cache = [-1] * (n + 1)
    cache[0], cache[1] = 1, 1

    for i in range(2, n + 1):
        cache[i] = cache[i - 1] + cache[i - 2]

    return cache[n]


# --------------- End of Fibonacci Algorithm --------------- #

# --------------- Change Making Algorithm --------------- #

# make change to down
def make_change(coins, amount):
    cache = [-1] * (amount + 1)
    return make_change_helper(coins, amount, cache)


def make_change_helper(coins, amount, cache):
    if amount == 0:
        return 0
    # set a min coins to max integer size
    min_coins = sys.maxsize
    # loop through each coin
    for i in coins:
        # if the coin itself is less than the amount then don't use this coin
        if i <= amount:
            # recursively call function with amount - i (coin)
            temp = make_change_helper(coins, amount - i, cache)
            # if temp is less than min_coins then we add 1 for the coin we subtracted, plus temp, because
            # perhaps that made further recursive calls, and so we need the coins required to solve for those amounts
            # as well. Now the reason we need temp to be greater than or equal to 0 is because of the case for a no
            # solution, since that will return a -1 we don't want to include that coin i made in this call into
            # min_coins, we want to skip it, just like we would skip if i <= amount, but since it was a possible
            # solution we did check it, only after the check did we find out it was not a solution if it returned
            # -1, so we skip it.
            if 0 <= temp < min_coins:
                min_coins = 1 + temp

    # so if all the coins would result in amount going to a negative number, then it means none
    # of the coins would solve for this amount, so we set cache at this amount to -1
    # the reason that this means this, is the for loop will change min_coins's value if the coin results in
    # an amount of 0 or more
    if min_coins == sys.maxsize:
        cache[amount] = -1
    else:
        cache[amount] = min_coins

    return cache[amount]


# NEED TO FINISH!!!!!!!
def make_change_bottom_up(coins, amount):
    cache = [-10] * (amount + 1)
    cache[0] = 0

    # loop through each amount number to solve for each starting from the bottom
    for i in range(amount):
        # for each amount number look at each coin
        if cache[i] != -10:
            continue
        min_val = float('infinity')
        for coin in coins:
            if i - coin >= 0:
                pass


# naive approach without memoization, and top down
def longest_common_subsequence_naive(str1, str2):
    return longest_common_subsequence_naive_helper(str1, str2, len(str1) - 1, len(str2) - 1)


def longest_common_subsequence_naive_helper(str1, str2, i, j):
    if i < 0 or j < 0:
        return 0

    if str1[i] == str2[j]:
        return 1 + longest_common_subsequence_naive_helper(str1, str2, i - 1, j - 1)
    else:
        return max(longest_common_subsequence_naive_helper(str1, str2, i - 1, j),
                   longest_common_subsequence_naive_helper(str1, str2, i, j - 1))


# approach with memoization also top down
def longest_common_subsequence(str1, str2):
    cache = [[-1 for i in range(len(str2))] for j in range(len(str1))]
    return longest_common_subsequence_helper(str1, str2, cache, len(str1) - 1, len(str2) - 1)


def longest_common_subsequence_helper(str1, str2, cache, i, j):
    if i < 0 or j < 0:
        return 0

    if str1[i] == str2[j]:
        cache[i][j] = 1 + longest_common_subsequence_helper(str1, str2, cache, i - 1, j - 1)
        return cache[i][j]
    else:
        cache[i][j] = max(longest_common_subsequence_helper(str1, str2, cache, i - 1, j),
                          longest_common_subsequence_helper(str1, str2, cache, i, j - 1))
        return cache[i][j]


# implement a bottom up approach still

# product sum
# I don't know how I came up with this, but drawing it out and going through the steps that
# I would use to solve this as a human really gave me the solution, normally though, I just think
# up a solution in my head, but realized drawing it out like this for more complex problems is
# super helpful in the design
def product_sum(numbers):
    # creates cache with size of number plus 1
    cache = [-1] * (len(numbers) + 1)
    # set's starting cache values, the 0 is needed as well as the extra + 1 size of the cache
    # because of cache[index - 2] in the helper function, if we used just the len(numbers) - 1
    # index, then at index 1, the cache[index - 2] would equal -1 since cache[-1] does not exist
    cache[0], cache[1] = 0, numbers[0]
    # calculates cache values
    product_sum_helper(numbers, len(numbers), cache)
    # return last cache value
    return cache[len(numbers)]


def product_sum_helper(numbers, index, cache):
    if cache[index] != -1:
        return
    # calculates cache values that will be used in following if statement
    product_sum_helper(numbers, index - 1, cache)
    # compares multiplying current numbers value to previous numbers value vs addition while accounting
    if numbers[index - 1] * numbers[index - 2] + cache[index - 2] \
            > numbers[index - 1] + cache[index - 1]:
        cache[index] = numbers[index - 1] * numbers[index - 2] + cache[index - 2]
    else:
        cache[index] = numbers[index - 1] + cache[index - 1]


# Still need to implement the rod algorithm

def knapsack(n, W, items, values):
    cache = [-1] * (W + 1)
    cache[0] = 0
    knapsack_helper(n, W, items, values, cache)
    return cache[W]


def knapsack_helper(n, W, items, values, cache):
    if cache[W] != -1:
        return cache[W]
    for i in range(len(items)):
        if W - items[i] > -1:
            temp = values[i] + knapsack_naive(n, W - items[i], items, values)
            if temp > cache[W]:
                cache[W] = temp
    return cache[W]


def knapsack_naive(n, W, items, values):
    max_val = 0
    for i in range(len(items)):
        if W - items[i] > -1:
            temp = values[i] + knapsack_naive(n, W - items[i], items, values)
            if temp > max_val:
                max_val = temp
    return max_val


# the perfected knapsack, pretty good code, this is the one from the lecture just for reference
def unbound_knapsack(W, n, weights, values):
    dp = [0] * (W + 1)
    for x in range(1, W + 1):
        for i in range(n):
            wi = weights[i]
            if wi <= x:
                dp[x] = max(dp[x], dp[x - wi] + values[i])
    return dp[W]


# so this basically creates a multidimensional array with length W + 1(weight of the backpack + 1), and
# width n + 1(number of items + 1). The idea is basically, start with a backpack that can carry a weight of 1, then
# for that backpack, what is the highest value it can get with only 1 item available, then what is the highest
# value it can get with 2 items, then 4 then 5, etc for however many items are there.
def knapsack_0_1(W, n, weights, values):
    dp = [[0 for x in range(n + 1)] for x in range(W + 1)]
    for x in range(1, W + 1):
        for i in range(1, n + 1):
            dp[x][i] = dp[x][i - 1]
            a = dp[x][i - 1]
            wi = weights[i - 1]
            vi = values[i - 1]
            if x >= wi:
                # the i - 1 part here is because we are now getting rid of one item, so if i was 4 for 4 items,
                # then we would go to the backpack of weight = x - wi, and then what was the best result for a
                # backpack of that size, and with now 3 items available.
                dp[x][i] = max(dp[x][i], dp[x - wi][i - 1] + vi)

    return dp[W][n]


# --------------- Binary Tree Class --------------- #

class BST:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


# --------------- Binary Tree Class --------------- #


# --------------- Branch Sums Algorithm --------------- #
# Returns an array of a binary trees path sums
def branchSums(root):
    results = []
    sums_helper(root, 0, results)
    return results


def sums_helper(node, sum, results):
    sum += node.value
    if node.left is None and node.right is None:
        results.append(sum)
        return

    if node.left is not None:
        sums_helper(node.left, sum, results)
    if node.right is not None:
        sums_helper(node.right, sum, results)


# --------------- Branch Sums Algorithm --------------- #


# --------------- Node Depths Algorithm --------------- #
# Returns the sum of all the path depths of a binary tree
def nodeDepths(root):
    result = [0]
    depth_helper(root, -1, result)
    return result[0]


def depth_helper(node, depth, result):
    if node is None:
        return

    r = node.value

    depth += 1
    result[0] += depth

    depth_helper(node.left, depth, result)
    depth_helper(node.right, depth, result)


def simplified_depth(root, depth=0):
    if root is None:
        return 0
    return depth + simplified_depth(root.left, depth + 1) + simplified_depth(root.right, depth + 1)


def set_up_tree():
    root = BST(1)
    root.left = BST(2)
    root.left.left = BST(4)
    root.left.right = BST(5)
    root.left.left.left = BST(8)
    root.left.left.right = BST(9)
    root.right = BST(3)
    root.right.left = BST(6)
    root.right.right = BST(7)
    return root

    # Driver For these methods
    # root = set_up_tree()
    # print(branchSums(root))
    # print(nodeDepths(root))
    # print(simplified_depth(root))


# --------------- Node Depths Algorithm --------------- #


class Node:
    def __init__(self, name):
        self.children = []
        self.name = name

    def addChild(self, name):
        self.children.append(Node(name))
        return self

    def depthFirstSearch(self, array):
        self.depth_search_helper(self, array)
        return array

    def depth_search_helper(self, node, array):
        array.append(node.name)
        r = node.children
        for i in node.children:
            # r = i.children[0]
            node.depth_search_helper(i, array)

    # Driver code
    # root = Node('A')
    # root.addChild('B')
    # root.addChild('C')
    # root.addChild('D')
    # root.children[0].addChild('E')
    # print(root.depthFirstSearch([]))


# --------------- Node Depths Algorithm --------------- #

# --------------- Min Query Wait Time Algorithm --------------- #

def minimumWaitingTime(queries):
    waiting_time = 0
    total_time = 0
    queries.sort()
    print(queries)
    if len(queries) > 1:
        for i in range(0, len(queries) - 1):
            waiting_time = waiting_time + queries[i]
            total_time += waiting_time
    return total_time

    # Driver Code
    # query = [3, 2, 1, 2, 6]
    # print(minimumWaitingTime(query))


# --------------- Min Query Wait Time Algorithm --------------- #

# --------------- Class Photos Algorithm --------------- #
def classPhotos(redShirtHeights, blueShirtHeights):
    redShirtHeights.sort()
    blueShirtHeights.sort()
    red = 0
    blue = 0
    for i in range(len(redShirtHeights)):
        if redShirtHeights[i] > blueShirtHeights[i]:
            red += 1
        elif redShirtHeights[i] < blueShirtHeights[i]:
            blue += 1

    if red == len(redShirtHeights) or blue == len(redShirtHeights):
        return True
    return False


# --------------- Class Photos Algorithm --------------- #


if __name__ == '__main__':
    print('Starting Program...')

