import random

for i in range(10):
    print('a{0} = {1}'.format(i, [random.randint(0, 1) for j in range(1000)]))

