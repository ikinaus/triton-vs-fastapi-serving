from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import re

RE_TITLE = re.compile(r" ([A-Za-z]+)\.")
SLASH_DOT = re.compile(r"[/.]")
FLAT_MAPPING = {
    "Mr": "Mr",
    "Miss": "Miss", "Mlle": "Miss", "Ms": "Miss",
    "Mrs": "Mrs", "Mme": "Mrs",
    "Master": "Master",
}

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, prefix_treshold=0.005) -> None:
        super().__init__()
        self.prefix_treshold = prefix_treshold
        self.ticket_frequency_ = {}
        self.rare_prefix_ = []

    @staticmethod
    def _missing(v) -> bool:
        return v is None or v == 'nan' or isinstance(v, float) and v != v

    def fit(self, X: pd.DataFrame, y = None) -> "FeatureExtractor":

        self.ticket_frequency_ = X['Ticket'].value_counts().to_dict()

        prefix_freq = (X['Ticket']
                        .str.replace(r'[/.]', '', regex=True)
                        .str.split()
                        .apply(lambda w: w[0] if len(w) > 1 else 'NoPrefix')
                        .value_counts(normalize=True))

        self.rare_prefix_ = prefix_freq[prefix_freq < self.prefix_treshold].index.to_list()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if 'PassengerId' in X.columns.to_list():
            X_out = X_out.drop(columns=['PassengerId'])

        cols = ['Cabin', 'Embarked']
        missing_mask = X_out[cols].isna() | (X_out[cols] == 'nan')
        X_out[cols] = X_out[cols].where(~missing_mask, np.nan).astype(object)
        X_out[['Age', 'Fare']] = X_out[['Age', 'Fare']].astype(float)

        # Cabin Feature
        X_out['HasCabin'] = X_out['Cabin'].notna().astype('int')
        X_out['Deck'] = X_out['Cabin'].str[0].fillna('U')
        X_out['CabinCount'] = ((X_out['Cabin']
                                .str.strip()
                                .str.count(' ')+1)
                                .fillna(0)
                                .astype('int'))
        X_out = X_out.drop(columns=['Cabin'])

        # Name Feature
        X_out['NameLen'] = X_out['Name'].str.len()
        X_out['Title'] = X_out['Name'].str.extract(pat=r' ([A-Za-z]+)\.', expand=False)

        X_out['Title'] = X_out['Title'].map(FLAT_MAPPING).fillna('Rare').astype(str)
        X_out = X_out.drop(columns=['Name'])

        # Group Size
        local_counts = X_out['Ticket'].value_counts().to_dict()
        X_out['GroupSize'] = X_out['Ticket'].apply(
            lambda t: max(self.ticket_frequency_.get(t, 1), local_counts.get(t, 1))
        )

        # TicketPrefix
        X_out['TicketPrefix'] = (X_out['Ticket']
                                    .str.replace(r'[/.]', '', regex=True)
                                    .str.split()
                                    .apply(lambda w: w[0] if len(w) > 1 else 'NoPrefix'))
        X_out['TicketPrefix'] = (X_out['TicketPrefix']
                                    .replace(self.rare_prefix_, 'Rare')
                                    .astype(str))
        X_out = X_out.drop(columns=['Ticket'])

        cat_cols = X_out.select_dtypes(include=['object', 'category']).columns
        X_out[cat_cols] = X_out[cat_cols].astype(str)

        return X_out

    def transform_online(self, rows: list[dict]) -> list[dict]:

        local_ticket_counter = {}
        for r in rows:
            t = r['Ticket']
            local_ticket_counter[t] = local_ticket_counter.get(t, 0) + 1

        out = []
        for r in rows:
            pclass, sibsp, parch, sex = r['Pclass'], r['SibSp'], r['Parch'], r['Sex']
            age = float(r['Age']) if r['Age'] is not None else None
            fare = float(r['Fare']) if r['Fare'] is not None else None

            cabin = r['Cabin']
            if self._missing(cabin):
                cabin = None

            embarked = r['Embarked']
            if self._missing(embarked):
                embarked = None

            if cabin is None:
                has_cabin, deck, cabin_count = 0, 'U', 0
            else:
                has_cabin = 1
                deck = cabin[0]
                cabin_count = cabin.strip().count(' ') + 1

            name = r['Name']
            name_len = len(name)
            m = RE_TITLE.search(name)
            title = FLAT_MAPPING.get(m.group(1), 'Rare') if m is not None else "Rare"

            # GroupSize, TicketPrefix.

            out.append({
                "Pclass": pclass,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Embarked": embarked,
                "HasCabin": has_cabin,
                "Deck": deck,
                "CabinCount": cabin_count,
                "NameLen": name_len,
                "Title" : title,
            })
        return out
