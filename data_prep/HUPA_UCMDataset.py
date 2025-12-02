from .BaseDataset import BaseDataset

class HUPA_UCMDataset(BaseDataset):
    def clean_cols(self):
        super().clean_cols()

        for i, df in enumerate(self.all_data):

            df["carbs"] = df["carbs"] * 10

            self.all_data[i] = df