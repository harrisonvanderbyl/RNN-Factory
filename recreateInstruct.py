import numpy as np

# tokens = np.load("instruct_data.npy")

# # save as instruct1.npy
# np.save("instruct1.npy", tokens[:len(tokens)//2])
# np.save("instruct2.npy", tokens[len(tokens)//2:])

# reconstruct
tokens = np.concatenate([np.load("instruct1.npy"), np.load("instruct2.npy")])

np.save("instruct_data.npy", tokens)