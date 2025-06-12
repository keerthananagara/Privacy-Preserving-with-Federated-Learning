import matplotlib.pyplot as plt

# Global epochs
epochs = list(range(1, 11))

# Accuracy values
acc_1_clients = [0.8069, 0.8280, 0.8405, 0.8468, 0.8501, 0.8549, 0.8590, 0.8607, 0.8631, 0.8669]
acc_5_clients = [0.5956, 0.7178, 0.7606, 0.7604, 0.7768, 0.7994, 0.7922, 0.7968, 0.8236, 0.8248]
acc_10_clients = [0.5480, 0.6919, 0.7340, 0.7433, 0.7666, 0.7777, 0.7965, 0.8058, 0.8008, 0.8106]

# Plotting Accuracy vs Epochs
plt.figure(figsize=(8, 5))
plt.plot(epochs, acc_1_clients, marker='o', label='1 Client', color='red')  # Added color
plt.plot(epochs, acc_5_clients, marker='o', label='5 Clients', color='blue') # Added color
plt.plot(epochs, acc_10_clients, marker='o', label='10 Clients', color='green') # Added color

plt.title('Accuracy vs Global Epochs')
plt.xlabel('Global Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()