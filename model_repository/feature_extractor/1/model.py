import os
import numpy as np
import triton_python_backend_utils as pb_utils
import pickle

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FIELDS = ("Pclass", "Name", "Sex", "Age", "SibSp", "Parch",
          "Ticket", "Fare", "Cabin", "Embarked")
STRING_FIELDS = {"Name", "Sex", "Ticket", "Cabin", "Embarked"}
OPTIONAL_FIELDS = {"Age", "Fare", "Cabin", "Embarked"}


class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        cur_path = os.path.dirname(os.path.abspath(__file__))
        extractor_pth = os.path.join(cur_path, 'extractor.pkl')

        with open(extractor_pth, "rb") as f:
            self.extractor = pickle.load(f)

    @staticmethod
    def _to_rows(request) -> list:
        tensors = {name: pb_utils.get_input_tensor_by_name(request, name)
                   for name in FIELDS}
        batch_size = tensors["Pclass"].as_numpy().shape[0]

        columns = {}
        for name in FIELDS:
            tensor = tensors[name]
            if tensor is None:
                if name not in OPTIONAL_FIELDS:
                    raise pb_utils.TritonModelException(
                        f"required input '{name}' is missing")
                columns[name] = [None] * batch_size
                continue

            values = tensor.as_numpy().flatten()
            columns[name] = ([v.decode("utf-8") for v in values]
                             if name in STRING_FIELDS else values.tolist())

        return [{name: columns[name][i] for name in FIELDS}
                for i in range(batch_size)]

    @staticmethod
    def _to_tensors(records: list) -> list:
        out = []
        for name in CATEGORICAL_FEATURES:
            column = np.array([r[name].encode("utf-8") for r in records],
                              dtype=object).reshape(-1, 1)
            out.append(pb_utils.Tensor(name, column))
        for name in NUMERIC_FEATURES:
            column = np.array([r[name] if r[name] is not None else np.nan
                               for r in records], dtype=np.float32).reshape(-1, 1)
            out.append(pb_utils.Tensor(name, column))
        return out

    def execute(self, requests: list) -> list:
        responses = []
        for request in requests:
            records = self.extractor.transform_online(self._to_rows(request))
            responses.append(
                pb_utils.InferenceResponse(output_tensors=self._to_tensors(records))
            )
        return responses
