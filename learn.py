import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Area (sq ft)': [1000, 1500, 2000, 2500, 3000],
    'Price (Rs Lakhs)': [50, 65, 80, 95, 110]
}

df = pd.DataFrame(data)

# Step 1: Split features (X) and labels (y)
X = df[['Area (sq ft)']]   # Features
y = df['Price (Rs Lakhs)']  # Labels

# Step 2: Create a model and train
model = LinearRegression()
model.fit(X, y)

# Step 3: Predict price for 2200 sq ft
predicted_price = model.predict([[2200]])
print(f"Predicted Price: Rs {predicted_price[0]:.2f} Lakhs")

# Step 4: Plotting
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (Rs Lakhs)")
plt.title("Linear Regression Example")
plt.legend()
plt.show()
