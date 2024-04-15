def pow1(x, y):
    "Calculate (x ** y) efficiently."
    number = 1
    while y:
        if y & 1:
            number = number * x
        y >>= 1
        x = x * x
    return number


def pow2(x, y):
    "Calculate (x ** y) efficiently."
    number = 1
    while True:
        if y & 1:
            number = number * x
        y >>= 1
        if not y:
            break
        x = x * x
    return number
print(pow1(2, 3))
print(pow2(2, 3))