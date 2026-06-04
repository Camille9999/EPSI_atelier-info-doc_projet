import pandas as pd
# cat encoding
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('data/food_menu_nutrition_dataset.csv')
df = df.drop(columns=['item_name', 'city','city_tier', 'is_bestseller'])

# print(df.columns.tolist())

# country = df['country'].unique()
# print("Country: ")
# print(country)
# brand_name = df['brand_name'].unique()
# print("Brand Name: ")
# print(brand_name)

bin_edges = [0, 3, 4, 5]
bin_labels = [1, 2, 3]
df['avg_rating'] = pd.cut(
    df['avg_rating'], 
    bins=bin_edges, 
    labels=bin_labels, 
    include_lowest=True
)
# print("Average Rating: ")
# print(df['avg_rating'].unique())

# print("item category: ")
# print(df['item_category'].unique())

# print("item subcategory: ")
# print(df['item_subcategory'].unique())

# print("brand tier: ")
# print(df['brand_tier'].unique())

df = df.drop(columns=['brand_name', 'item_category'])

encoder = OneHotEncoder(sparse_output=False,
                        feature_name_combiner=lambda col, category: str(category)) 
categorical_cols = ['item_subcategory', 'brand_tier']
encoded_array = encoder.fit_transform(df[categorical_cols])

clean_column_names = encoder.get_feature_names_out(
    input_features=categorical_cols
)

encoded_cols_df = pd.DataFrame(
    data=encoded_array, 
    columns=clean_column_names,
    index=df.index
)

df = df.drop(columns=categorical_cols)
df = pd.concat([df, encoded_cols_df], axis=1)

def make_csvs(df, country):
    df_country = df[df['country'] == country]
    df_country = df_country.drop(columns=['country'])
    df_country.to_csv(f'data/data_{country.lower()}.csv', index=False)
make_csvs(df, 'India')
make_csvs(df, 'USA')
make_csvs(df, 'UK')
make_csvs(df, 'Canada')
make_csvs(df, 'Australia')

