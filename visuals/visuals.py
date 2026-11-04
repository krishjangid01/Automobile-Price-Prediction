import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_automobile.csv")

plt.figure(figsize=(8, 5))
plt.scatter(df["engine-size"], df["price"])
plt.xlabel("Engine Size")
plt.ylabel("Automobile Price")
plt.title("Engine Size vs Automobile Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("engine_size_vs_price.png", dpi=300)
plt.show()
