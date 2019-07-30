from PIL import Image
import torch
import torch.utils.data as D
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
import pandas as pd


class ImagesDS(D.Dataset):
    def __init__(
        self,
        df,
        img_dir,
        transform=None,
        mode="train",
        site=1,
        channels=[1, 2, 3, 4, 5, 6],
    ):

        self.transform = transform
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]

    @staticmethod
    def _load_img_as_tensor(file_name, transform=None):
        with Image.open(file_name) as img:
            if transform:
                img = transform(img)
            return T.ToTensor()(img)

    def _get_img_path(self, index, channel):
        experiment, well, plate = (
            self.records[index].experiment,
            self.records[index].well,
            self.records[index].plate,
        )
        return "/".join(
            [
                self.img_dir,
                self.mode,
                experiment,
                f"Plate{plate}",
                f"{well}_s{self.site}_w{channel}.png",
            ]
        )

    def __getitem__(self, index):
        paths = [self._get_img_path(index, ch) for ch in self.channels]
        img = torch.cat(
            [
                self._load_img_as_tensor(img_path, transform=self.transform)
                for img_path in paths
            ]
        )

        if self.mode == "train":
            return img, self.records[index].sirna
        else:
            return img, self.records[index].id_code

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


def prepare_data_ds(path_data):
    df = pd.read_csv(path_data + "/train.csv")
    df_train, df_val = train_test_split(
        df, test_size=0.05, stratify=df.sirna, random_state=42
    )
    df_test = pd.read_csv(path_data + "/test.csv")

    ds = ImagesDS(df_train, path_data, mode="train")
    ds_val = ImagesDS(df_val, path_data, mode="train")
    ds_test = ImagesDS(df_test, path_data, mode="test")
    return ds, ds_val, ds_test, df_train, df_val, df_test
