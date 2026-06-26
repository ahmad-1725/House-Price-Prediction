import pandas as pd
import numpy as np

df = pd.read_csv('data/zameen-updated.csv')

df = df.drop(columns = [
  "property_id",
  "location_id",
  "page_url",
  "date_added",
  "agency",
  "agent",
])

# keeping only for sale houses
df = df[df["purpose"]=="For Sale"]

# keeping only valid entries
df = df[df["price"] > 0]
df = df[df["Area Size"] > 0]
df = df[df["bedrooms"] > 0]
df = df[df["baths"] > 0]

# selecting features and label
x = df[[
  "property_type",
  "location",
  "city",
  "province_name",
  "latitude",
  "latitude",
  "baths",
  "area",
  "purpose",
  "bedrooms",
  "Area Type",
  "Area Size"
]]

y = df["price"]

