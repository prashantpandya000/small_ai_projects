import matplotlib.pyplot as plt
import neurolab as nl

input=[[0,0],[0,1],[1,0],[1,1]]
target=[[0],[0],[0],[0]]

net=nl.net.newp([[0,1],[0,1]],1)

error_progress=net.train(input,target, epochs=100, show=10, lr=0.1)

plt.figure()
plt.plot(error_progress)
plt.xlabel('number of epochs')
plt.ylabel('training error')
plt.grid()
plt.show()