import numpy as np 
def f(x):
    ans = (x%6)**2%7 - np.sin(x)
    return ans


def f2(x):
    ans = -x**4 + 1000*x**3 - 20*x**2 +4*x - 6
    return ans

best_x = 0
higher = 0
for x in range(1,101):
    #print(x)
    val = f(x)
    #print(val)
    #print('{}    {}'.format(val,higher))
    if val > higher:
        higher = val
        best_x = x

print('for the first equation: best x={}  max val = {}'.format(best_x,higher))

best_x = 0
higher = 0
for x in range(1,999):
    val = f2(x)
    #print('{}    {}'.format(val,higher))
    if val > higher:
        higher = val
        best_x = x

print('for the second equation: best x={}  max val = {}'.format(best_x,higher))
