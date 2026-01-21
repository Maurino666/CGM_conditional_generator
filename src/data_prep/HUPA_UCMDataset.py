from .SequenceableDataset import SequenceableDataset


class HUPA_UCMDataset(SequenceableDataset):
    def _clean_cols(self):
        super()._clean_cols()

        for i, df in enumerate(self.all_data):

            df["carbs"] = df["carbs"] * 10

            self.all_data[i] = df