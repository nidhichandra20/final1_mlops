import joblib
import pandas as pd

model = joblib.load("model.pkl")

sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
)

pred = model.predict(sample)

assert pred[0] in [0, 1, 2]
print("Model test passed")
