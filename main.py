import sys
import copy


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


# --------------- Tandem Bicycle Algorithm --------------- #

def tandemBicycle(redShirtSpeeds, blueShirtSpeeds, fastest):
    blueShirtSpeeds.sort()

    if fastest:
        redShirtSpeeds.sort(reverse=True)
    else:
        redShirtSpeeds.sort()

    top_speed = 0

    for i in range(len(redShirtSpeeds)):
        top_speed += max(redShirtSpeeds[i], blueShirtSpeeds[i])

    return top_speed


# --------------- Tandem Bicycle Algorithm --------------- #

# --------------- Remove Duplicates Algorithm --------------- #

class LinkedList:
    def __init__(self, value):
        self.value = value
        self.next = None


def removeDuplicatesFromLinkedList(linkedList):
    temp_node = linkedList

    while temp_node.next is not None:
        r = temp_node.value
        p = temp_node.next.value
        if temp_node.value == temp_node.next.value:
            temp_node.next = temp_node.next.next
        else:
            temp_node = temp_node.next
    return linkedList

    # Driver Code
    # l = LinkedList(1)
    # l.next = LinkedList(1)
    # l.next.next = LinkedList(3)
    # l.next.next.next = LinkedList(4)
    # l.next.next.next.next = LinkedList(4)
    # l.next.next.next.next.next = LinkedList(4)
    # l.next.next.next.next.next.next = LinkedList(5)
    # l.next.next.next.next.next.next.next = LinkedList(6)
    # l.next.next.next.next.next.next.next.next = LinkedList(6)
    # print(removeDuplicatesFromLinkedList(l))


# --------------- Remove Duplicates Algorithm --------------- #

# --------------- Fib Algorithm --------------- #
def getNthFib(n):
    cache = [-1] * (n + 1)
    cache[0], cache[1] = 0, 1
    return fib_helper(n - 1, cache)


def fib_helper(n, cache):
    if cache[n] != -1:
        return cache[n]
    cache[n] = fib_helper(n - 1, cache) + fib_helper(n - 2, cache)
    return cache[n]


# --------------- Fib Algorithm --------------- #

# --------------- Product Sum Algorithm --------------- #
def productSum(array):
    return product_helper(array, 0)


def product_helper(num, multiplier):
    calc_sum = 0
    if type(num) == list:
        multiplier += 1
        for i in range(len(num)):
            calc_sum += product_helper(num[i], multiplier)
        calc_sum = calc_sum * multiplier
    else:
        calc_sum = num
    return calc_sum

    # Driver Code
    # r = [5, 2, [7, -1], 3, [6, [-13, 8], 4]]
    # print(productSum(r))


# --------------- Product Sum Algorithm --------------- #

# --------------- Insertion Sort Algorithm --------------- #

def insertionSort(array):
    for i in range(1, len(array)):
        if array[i - 1] > array[i]:
            array[i - 1], array[i] = array[i], array[i - 1]
            for j in reversed(range(1, i)):
                if array[j] < array[j - 1]:
                    array[j], array[j - 1] = array[j - 1], array[j]
                else:
                    break
    return array

    # Driver Code
    # a = [8, 5, 2, 9, 5, 6, 3]
    # print(insertionSort(a))


# --------------- Insertion Sort Algorithm --------------- #

# --------------- Selection Sort Algorithm --------------- #
def selectionSort(array):
    # Write your code here.

    for i in range(len(array)):
        min = array[i]
        index = i
        for j in range(i + 1, len(array)):
            if array[j] < min:
                min = array[j]
                index = j
        array[i], array[index] = min, array[i]
    return array


# --------------- Selection Sort Algorithm --------------- #

# --------------- Palindrome Check Algorithm --------------- #
def isPalindrome(string):
    idx = len(string) - 1
    for i in range(len(string)):
        if string[i] != string[idx]:
            return False
        idx -= 1
    return True

    # Driver Code
    # s = 'abcdcba'
    # print(isPalindrome(s))


# --------------- Palindrome Check Algorithm --------------- #

# --------------- Caesar Cipher Algorithm --------------- #

def caesarCipherEncryptor(string, key):
    new_string = ''

    for i in range(len(string)):
        num = ord(string[i]) + (key % 26)

        if num > 122:
            num -= 26
        new_string += chr(num)
    return new_string

    # Driver Code
    # print(caesarCipherEncryptor('abc', 52))


# --------------- Caesar Cipher Algorithm --------------- #

# --------------- Run Length Encoding Algorithm --------------- #

def runLengthEncoding(string):
    # Write your code here.
    new_string = ''
    start = string[0]
    num = 0
    for i in string:

        if i == start:
            num += 1

        else:
            new_string += str(num) + start
            start = i
            num = 1

        if num == 10:
            new_string += '9' + start
            num = 1

    new_string += str(num) + start

    return new_string

    # Driver Code
    # print(runLengthEncoding("AAAAAAAAAAAAABBCCCCDD"))


# --------------- Run Length Encoding Algorithm --------------- #

# --------------- Generate Document Algorithm --------------- #

def generateDocument(characters, document):
    cache = {}
    for i in characters:
        if i not in cache:
            cache[i] = 1
        else:
            cache[i] += 1

    for j in document:
        if j not in cache or cache[j] == 0:
            return False
        else:
            cache[j] -= 1
    return True

    # Driver Code
    # print(generateDocument("aheaolabbhb", "hello"))


# --------------- Generate Document Algorithm --------------- #

# --------------- First Non-Repeating Character Algorithm --------------- #

def firstNonRepeatingCharacter(string):
    str_array = list(string)
    visited = []

    index = 0
    while 0 != len(str_array):
        val = str_array.pop(0)
        if val not in str_array and val not in visited:
            return index
        visited.append(val)
        index += 1
    return -1

    # Driver Code
    # s = "faadabcbbebdf"
    # print(firstNonRepeatingCharacter(s))


# --------------- First Non-Repeating Character Algorithm --------------- #

# --------------- Three Number Sum Algorithm --------------- #

def threeNumberSum(array, targetSum):
    result = []
    three_sum_helper(array, targetSum, [], result, 0)
    result.sort()
    return result


def three_sum_helper(array, target_sum, choices, result, index):
    if len(choices) == 3:
        if sum(choices) == target_sum and choices not in result:
            temp = copy.deepcopy(choices)
            temp.sort()
            result.append(temp)
        return

    for i in range(index, len(array)):
        choices.append(array[i])
        three_sum_helper(array, target_sum, choices, result, i + 1)
        choices.pop()

    # Driver Code
    # a = [1, 1, 1, 2, 3, 5, 3]
    # print(threeNumberSum(a, 3))


# --------------- Three Number Sum Algorithm --------------- #


# --------------- Smallest Difference Algorithm --------------- #
# n * m = around n^2 time complexity
def mySmallestDifference(arrayOne, arrayTwo):
    min = float('infinity')
    results = [0, 0]
    for i in arrayOne:
        for j in arrayTwo:
            if abs(i - j) < min:
                min = abs(i - j)
                results[0], results[1] = i, j

    return results


# nlogn + mlogm time complexity
def smallestDifference(arrayOne, arrayTwo):
    arrayOne.sort()
    arrayTwo.sort()
    idx1 = 0
    idx2 = 0
    smallest = float('inf')
    current = float('inf')
    smallestPair = []
    while idx1 < len(arrayOne) and idx2 < len(arrayTwo):
        first = arrayOne(idx1)
        second = arrayTwo(idx2)

        if first < second:
            current = second - first
            idx1 += 1
        elif second < first:
            current = first - second
            idx2 += 1
        else:
            return [first, second]
        if smallest > current:
            smallest = current
            smallestPair = [first, second]
    return smallestPair

    # when looking at -1 in a and comparing to 15, since -1 is less than 15
    # it wouldn't make sense to compare -1 to a larger number than 15 since that would result
    # in a larger difference, so we know that we should increment idx1 since the difference of -1 and 15
    # is known to be the smallest, and we continue this way. That is why we sort the two arrays so we know that
    # the next value is always larger than the current ones we are looking at.
    # a = [-1, 3, 5, 10, 20, 28]
    # b = [15, 17, 26, 134, 135]


# --------------- Smallest Difference Algorithm --------------- #

# --------------- Move Element To End Algorithm --------------- #

def moveElementToEnd(array, toMove):
    idx1 = 0
    idx2 = len(array) - 1

    while idx1 < idx2:
        if array[idx1] != toMove and array[idx2] == toMove:
            idx1 += 1
            idx2 -= 1
        elif array[idx1] == toMove and array[idx2] == toMove:
            idx2 -= 1
        elif array[idx1] != toMove and array[idx2] != toMove:
            idx1 += 1
        else:
            array[idx1], array[idx2] = array[idx2], array[idx1]
    return array

    # Driver Code
    # a = [2, 1, 2, 2, 2, 3, 4, 2]
    # b = 2


# --------------- Move Element To End Algorithm --------------- #

# --------------- Monotonic Algorithm --------------- #

def isMonotonic(array):
    increasing = True
    decreasing = True

    for i in range(len(array) - 1):

        if array[i] < array[i + 1]:
            increasing = False
        if array[i] > array[i + 1]:
            decreasing = False

    return increasing or decreasing

    # Driver Code
    # a = [1, 1, 1, 1, 1, 1, 1]
    # print(isMonotonic(a))


# --------------- Monotonic Algorithm --------------- #

# --------------- Spiral Traverse Algorithm --------------- #

def spiralTraverse(array):
    total_n = len(array) * len(array[0])
    route = []
    col = [0, len(array[0]) - 1]
    rows = [1, len(array) - 2]
    while total_n > len(route):

        for i in range(col[0], col[1] + 1):
            route.append(array[col[0]][i])

        if total_n == len(route):
            break

        for i in range(rows[0], rows[1] + 1):
            route.append(array[i][col[1]])

        for i in reversed(range(col[0], col[1] + 1)):
            route.append(array[rows[1] + 1][i])

        if total_n == len(route):
            break

        for i in reversed(range(rows[0], rows[1] + 1)):
            route.append(array[i][col[0]])

        col[0] += 1
        col[1] -= 1
        rows[0] += 1
        rows[1] -= 1

    return route

    # Driver Code
    # a = [
    #    [1, 2, 3],
    #    [12, 13, 4],
    #    [11, 14, 5],
    #    [10, 15, 6],
    #    [9, 8, 7]
    # ]
    # print(spiralTraverse(a))


# --------------- Spiral Traverse Algorithm --------------- #


# --------------- Longest Peak Algorithm --------------- #
def longestPeak(array):
    count = 0
    increasing = 0
    decreasing = 0
    peak = 0

    for i in range(len(array) - 1):
        if array[i] < array[i + 1]:
            if increasing == 1 and decreasing == 1 and count > peak:
                peak = count
                count = 1
                increasing = 1
                decreasing = 0
            elif increasing == 0:
                increasing = 1
                count += 1

            count += 1

        elif array[i] > array[i + 1] and increasing == 1:
            decreasing = 1
            count += 1
        else:
            if increasing == 1 and decreasing == 1 and count > peak:
                peak = count
            count = 0
            increasing = 0
            decreasing = 0
    if count > peak and increasing == 1 and decreasing == 1:
        peak = count
    return peak


# --------------- Longest Peak Algorithm --------------- #

# --------------- Array Of Products Algorithm --------------- #
# very interesting structure, solved this with recursion by passing forward, the multiplication
# of previous values, and passing backwards the multiplication of current and future values, I guess.
def arrayOfProducts(array):
    results = [0] * len(array)
    results[0] = array_helper(array, 1, results, array[0])
    return results


def array_helper(array, index, results, value):
    if index == len(array):
        return 1
    val = array_helper(array, index + 1, results, array[index] * value)
    results[index] = value * val
    return array[index] * val


# --------------- Array Of Products Algorithm --------------- #


if __name__ == '__main__':
    print('Starting Program...')
    # Driver Code
    a = [1, 2, 3, 4, 5, 6, 10, 100, 1000]
    print(longestPeak(a))
