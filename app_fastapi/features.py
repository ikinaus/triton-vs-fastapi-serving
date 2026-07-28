from typing import Any, Self
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, prefix_treshold=0.005) -> None:
        super().__init__()
        self.prefix_treshold = prefix_treshold
        self.ticket_frequency_ = {}
        self.rare_prefix_ = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        
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

        flat_mapping = {
            'Mr': 'Mr',
            'Miss': 'Miss', 'Mlle': 'Miss', 'Ms': 'Miss',
            'Mrs': 'Mrs', 'Mme': 'Mrs',
            'Master': 'Master'
        }
        X_out['Title'] = X_out['Title'].map(flat_mapping).fillna('Rare').astype(str)
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
