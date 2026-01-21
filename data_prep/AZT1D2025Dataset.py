from .BaseDataset import BaseDataset


class AZT1D2025Dataset (BaseDataset):
    def clean_cols(self):
        super().clean_cols()
        for i, df in enumerate(self.all_data):

            df["device_mode"] = df["device_mode"].replace({"0": "Unknown", 0: "Unknown"}).fillna("Unknown").astype("category")

            df["bolus_type"] = df["bolus_type"].replace({"0": "None", 0: "None"}).fillna("None").astype("category")

            df["basal_rate"] = df["basal_rate"].ffill()

            self.all_data[i] = df