import numpy as np
import nn


x = np.random.rand(20).reshape(-1, 4)
y = np.round(np.random.rand(5))

net = nn.NNClassifier(5, 7)
net.fit(x, y)

res = [net.predict(xs)[0] for xs in x]

for r in res:
    print(r)

print('Test sat!')
