```python
import pandas as pd

# Create a DataFrame with specific values
fruits = pd.DataFrame({
    'Apples': [30],
    'Bananas': [21]
})

# Create a DataFrame with custom index labels
fruit_sales = pd.DataFrame(
    {
        'Apples': [35, 41],
        'Bananas': [21, 34]
    },
    index=['2017 Sales', '2018 Sales']
)

# Create a Series
ingredients = pd.Series(
    ['4 cups', '1 cup', '2 large', '1 can'],
    index=['Flour', 'Milk', 'Eggs', 'Spam'],
    name='Dinner'
)

# Read a CSV file
reviews = pd.read_csv(
    "../input/wine-reviews/winemag-data_first150k.csv",
    index_col=0
)

# Select one column
desc = reviews['description']

# Select the first description using loc
first_description = reviews.loc[0, 'description']

# Select the first row using iloc
first_row = reviews.iloc[0]

# Select the first 10 descriptions
first_descriptions = reviews.loc[:9, 'description']

# Select rows with specific index values
sample_reviews = reviews.loc[reviews.index.isin([1, 2, 3, 5, 8])]

# Select specific rows and columns using iloc
df = reviews.iloc[
    [0, 1, 10, 100],
    [
        reviews.columns.get_loc('country'),
        reviews.columns.get_loc('province'),
        reviews.columns.get_loc('region_1'),
        reviews.columns.get_loc('region_2')
    ]
]

# Select the country and variety columns of the first 100 records
df = reviews.iloc[
    :100,
    [
        reviews.columns.get_loc('country'),
        reviews.columns.get_loc('variety')
    ]
]

# Filter Italian wines
italian_wines = reviews.loc[reviews.country == 'Italy']

# Filter high-scoring wines from Australia and New Zealand
top_oceania_wines = reviews.loc[
    (reviews.country.isin(['Australia', 'New Zealand'])) &
    (reviews.points >= 95)
]

# Calculate median points
median_points = reviews.points.median()

# Get unique countries
countries = reviews.country.unique()

# Count reviews per country
reviews_per_country = reviews.country.value_counts()

# Center the price column around the mean price
centered_price = reviews.price - reviews.price.mean()

# Display summary statistics for price
price_summary = reviews.price.describe()

# Find the wine with the highest points-to-price ratio
bargain_idx = (reviews['points'] / reviews['price']).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']

# Count how many descriptions contain specific words
n_tropical = reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()

descriptor_counts = pd.Series(
    [n_tropical, n_fruity],
    index=["tropical", "fruity"]
)

# Convert points into star ratings
star_ratings = reviews.points.map(
    lambda x: 3 if x >= 95 else 2 if x >= 85 else 1
)

print("Coursework completed successfully.")
```
