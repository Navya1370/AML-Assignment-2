import os

print("\n========== RUNNING ALL MODELS ==========\n")

print("\n--- ID3 ---\n")
os.system("python id3.py")

print("\n--- C4.5 ---\n")
os.system("python c45.py")

print("\n--- CART ---\n")
os.system("python cart.py")

print("\n--- SKLEARN MODELS ---\n")
os.system("python sklearn_models.py")

print("\n========== DONE ==========\n")