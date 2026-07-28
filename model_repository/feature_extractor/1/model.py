import os
import numpy as np
import pandas as pd
import triton_python_backend_utils as pb_utils
import pickle

class TritonPythonModel:
    def initialize(self, args:dict) -> None:
        cur_path = os.path.dirname(os.path.abspath(__file__))
        extractor_pth = os.path.join(cur_path, 'extractor.pkl')

        with open(extractor_pth, "rb") as f:
            self.extractor = pickle.load(f)

    def execute(self, requests:list) -> list:
        responses = []

        FIELDS = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch",
          "Ticket", "Fare", "Cabin", "Embarked"]
        STRING_FIELDS = {"Name", "Sex", "Ticket", "Cabin", "Embarked"}
        OPTIONAL_FIELDS = {"Age", "Fare", "Cabin", "Embarked"}

        categorical_features = ['Sex', 'Embarked', 'Deck', 'TicketPrefix', 'Title']

        for request in requests:
            tensors = {
                name: pb_utils.get_input_tensor_by_name(request, name) for name in FIELDS
            }
            batch_size = tensors['Pclass'].as_numpy().shape[0]

            row = {}
            for name in FIELDS:
                if name in OPTIONAL_FIELDS:
                    if tensors[name] is None:
                        if name not in STRING_FIELDS:
                            row[name] = np.full(shape=batch_size, fill_value=np.nan)
                            continue
                        else:
                            row[name] = np.full(shape=batch_size, fill_value=np.nan, dtype=object)
                            continue

                vals = tensors[name].as_numpy().flatten()
                row[name] = [v.decode('utf-8') for v in vals] if name in STRING_FIELDS else vals

            df = pd.DataFrame.from_dict(row)
            df_prep = self.extractor.transform(df)

            for col in df_prep.columns:
                if col in categorical_features:
                    df_prep[col] = df_prep[col].str.encode('utf-8')
                else:
                    df_prep[col] = df_prep[col].astype('float32')

            out_tensors = [
                pb_utils.Tensor(name, df_prep[name].values.reshape(-1, 1)) 
                for name in df_prep.columns
            ]
            inference = pb_utils.InferenceResponse(output_tensors=out_tensors)
            responses.append(inference)
        
        return responses