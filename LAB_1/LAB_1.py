# Gautam Kumar Mahar 
# 2103114 
# Machine Learning LAB 1
#------------------

import numpy as np

np.random.seed(100)
input_data = np.random.rand(1, 3)  
target_output = np.random.rand(1, 1)  

weights = np.random.rand(3, 1)  
bias = np.random.rand(1, 1)  

# y = wx + b
predicted_output = np.dot(input_data, weights) + bias

#  (y' - y)**2  / N;
forward_pass_loss = 0.5 * np.sum((predicted_output - target_output) ** 2) / input_data.shape[0]

print("Forward Pass Loss:", forward_pass_loss)

#  dl/dw

dL_dw = np.dot(input_data.T, (predicted_output - target_output)) / input_data.shape[0]

#  dl_db
dL_db = np.sum(predicted_output - target_output) / input_data.shape[0]



h = 0.001
weights[2] -= h # reduce a Miner(0.001) value from each weight 

predicted_output = np.dot(input_data, weights) + bias
#  (y' - y)**2  / N;
new_forward_pass_loss = 0.5 * np.sum((predicted_output - target_output) ** 2) / input_data.shape[0]

#  new dl_db
New_dL_db = 2* np.sum(predicted_output - target_output) / input_data.shape[0]

# new_forward_pass_loss = forward_pass_loss
print("New Forward Pass Loss:", new_forward_pass_loss)


predicted_output = np.dot(input_data, weights) + bias

# df/dx = f(x, y+h) - f(x,y)/h
backward_pass_loss = 0.5 * np.sum((predicted_output - target_output) ** 2) / input_data.shape[0]

# Compare forward pas
# print(dl_dw - backward_pass_loss / h)

print("Backward Pass Loss:", backward_pass_loss)
print("Gradient of weights:\n", dL_dw)
print("New Gradient of weight \n", (forward_pass_loss - new_forward_pass_loss) / h )
print("Gradient of bias:", dL_db)
# print("New Gradient of bias:",  (New_dL_db - dL_db  ))
print("New NGradient of bias:", (  New_dL_db -dL_db) / h)




 
