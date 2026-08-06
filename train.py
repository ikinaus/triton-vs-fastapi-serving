from pathlib import Path
import sys 

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'Data'
ARTIFACTS_DIR = ROOT / 'artifacts'
SRC_DIR = ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

from features import FeatureExtractor

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier

from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm

import pickle


numeric_features = ['Pclass', 'Age', 'SibSp',
                    'Parch', 'Fare', 'HasCabin',
                    'CabinCount', 'GroupSize', 'NameLen']
categorical_features = ['Sex', 'Embarked', 'Deck', 'TicketPrefix', 'Title']
lgb_params = {
    'learning_rate': 0.1, 
    'max_depth': 4, 
    'min_child_samples': 30, 
    'n_estimators': 200, 
    'num_leaves': 7
}

# Making the Pipeline
numeric_transformer = 'passthrough'
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values='nan', strategy='constant', fill_value='missing')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

best_lgb = LGBMClassifier(
    **lgb_params,
    random_state=42,
    verbose=-1
)

pipeline = Pipeline(steps=[
    ('extractor', FeatureExtractor()),
    ('preprocessor', preprocessor),
    ('classifier', best_lgb)
])

# Reading and splitting data
train_df = pd.read_csv(DATA_DIR / 'train.csv')
y = train_df['Survived']
X = train_df.drop(columns=['Survived'])
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train the model
pipeline.fit(X_train, y_train)

# ONNX conversion preprocessing
update_registered_converter(
    LGBMClassifier, 
    'LightGbmLGBMClassifier',
    calculate_linear_classifier_output_shapes, 
    convert_lightgbm,
    options={"nocl": [True, False], "zipmap": [True, False, "columns"]}
)

onnx_pipeline = Pipeline([
    ('preprocessor', pipeline.named_steps['preprocessor']),
    ('classifier', pipeline.named_steps['classifier'])
])

initial_types = [
    ('Pclass', FloatTensorType([None, 1])),
    ('Age', FloatTensorType([None, 1])),
    ('SibSp', FloatTensorType([None, 1])),
    ('Parch', FloatTensorType([None, 1])),
    ('Fare', FloatTensorType([None, 1])),
    ('HasCabin', FloatTensorType([None, 1])),
    ('CabinCount', FloatTensorType([None, 1])),
    ('GroupSize', FloatTensorType([None, 1])),
    ('NameLen', FloatTensorType([None, 1])),
    ('Sex', StringTensorType([None, 1])),
    ('Embarked', StringTensorType([None, 1])),
    ('Deck', StringTensorType([None, 1])),
    ('TicketPrefix', StringTensorType([None, 1])),
    ('Title', StringTensorType([None, 1]))
]

onnx_model = convert_sklearn(
    onnx_pipeline,
    initial_types=initial_types,
    target_opset={'': 12, 'ai.onnx.ml': 3},
    options={'zipmap': False}
)

# ONNX conversion
with open(ARTIFACTS_DIR / 'model.onnx', 'wb') as f:
    f.write(onnx_model.SerializePartialToString())

# Making the pkl file (Serializing the FeatureExtractor object)
with open(ARTIFACTS_DIR / 'extractor.pkl', 'wb') as f:
    pickle.dump(pipeline.named_steps['extractor'], f)