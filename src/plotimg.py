import matplotlib.pyplot as plt

C=[1/2*229405205/7,1/4*229405205/7,1/8*229405205/7,1/16*229405205/7,1/32*229405205/7]
# C=[1/2*229405205/7,1/4*229405205/7,1/8*229405205/7,1/16*229405205/7,1/32*229405205/7,1/48*229405205/7,1/64*229405205/7,1/128*229405205/7]
result = [1e-4,5e-4,8e-4,1e-3,1.5e-3]
rctr = 0.008


import numpy as np
import math
from scipy.optimize import curve_fit

x = np.array(C)
y = np.array(result)
params = np.array([1,1])

# print(5.29166666e-04+rctr*6.30445854e+04/(1/32*229405205/7))
for i in range(len(C)):
    # print(5.29166666e-4+rctr*6.30445854e+4/(C[i]))
    # print(-4.18404795e-03*rctr + 9.31356664e+00/math.log(C[i]))
    # print(0.08520938/math.log(C[i]) - 0.60997488*rctr)
    print(0.1060081/math.log(C[i]) -0.77394169*rctr)   # [1e-4,6e-4,8e-4,1e-3,1.5e-3]


def funcinv(x, a, b):
    return b*rctr + a/np.log(x)


res = curve_fit(funcinv, x, y, params)
print(res)

# plt.title('Threshold curve (rctr=0.0080)')
# plt.xlabel('the size of budget')
# plt.ylabel('Threshold')
# plt.plot(C, [-2.47439323e-03+ rctr * 6.03157148/math.log(C[i]) for i in range(len(C))])
# plt.show()