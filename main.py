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
    if numbers[index - 1] * numbers[index - 2] + cache[index - 2]\
            > numbers[index - 1] + cache[index - 1]:
        cache[index] = numbers[index - 1] * numbers[index - 2] + cache[index - 2]
    else:
        cache[index] = numbers[index - 1] + cache[index - 1]


if __name__ == '__main__':
    # print(fibonacci(7))
    # print(fibonacci_bottom_up_naive(7))
    # print(fibonacci_bottom_up(7))
    # coin_array = [1, 2, 5]
    # print(make_change(coin_array, 11))
    # print(make_change(coin_array, 5))
    # word1 = 'pozysinksi'
    # word2 = 'swozoniky'
    # print(longest_common_subsequence_naive(word1, word2))
    print(product_sum([2, 2, 1, 3, 2, 1, 2, 2, 1]))
