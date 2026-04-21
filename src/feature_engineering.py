from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def engineer_behavioral_features(df):
  df_engineered = df.assign(
      Admin_Ratio = df['Administrative'] / (df['Administrative'] + df['ProductRelated'] + df['Informational'] + 1),
      Info_Ratio =  df['Informational'] / (df['Administrative'] + df['ProductRelated'] + df['Informational'] + 1),
      Product_Efficiency = df['ProductRelated_Duration'] / (df['ProductRelated'] + 1),
      Is_High_Urgency = df['SpecialDay'].apply(lambda x: 1 if x > 0.8 else 0)
  )
  return df_engineered


