import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/zameen-updated.csv")

df = df.drop(
    columns=[
        "property_id",
        "location_id",
        "page_url",
        "date_added",
        "agency",
        "agent",
    ]
)

# keeping only for sale houses
df = df[df["purpose"] == "For Sale"]

# keeping only valid entries
df = df[df["price"] > 0]
df = df[df["Area Size"] > 0]
df = df[df["bedrooms"] > 0]
df = df[df["baths"] > 0]

# selecting features and label
x = df[
    [
        "property_type",
        "location",
        "city",
        "province_name",
        "latitude",
        "longitude",
        "baths",
        "area",
        "purpose",
        "bedrooms",
        "Area Type",
        "Area Size",
    ]
]

# target label
y = df["price"]


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.2, random_state=42
)

categorical_cols = [
    "property_type",
    "city",
    "province_name",
    "location",
    "Area Type",
]

numeric_cols = ["latitude", "longitude", "baths", "bedrooms", "Area Size"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
        ),
    ]
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE : ", mae)
print("RMSE : ", rmse)
print("R2 : ", r2)



joblib.dump(model, "Zameen.Com-House-Price-Model.pkl")

print("Saved Successfully")
