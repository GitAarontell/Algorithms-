import sys


# --------------- Fibonacci Algorithm --------------- #
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

# def make_change(coins, amount):
#    return make_change_helper(coins, amount)


def make_change(coins, amount):
    cache = [-1] * (amount + 1)
    make_change_helper(coins, amount, cache)
    return cache[amount]


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
            # solution, since that will return a -1
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


if __name__ == '__main__':
    print(fibonacci(7))
    print(fibonacci_bottom_up_naive(7))
    print(fibonacci_bottom_up(7))
    coin_array = [2]
    print(make_change(coin_array, 3))
    print(make_change(coin_array, 4))
