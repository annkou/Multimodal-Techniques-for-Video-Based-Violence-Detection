import csv
import urllib.parse

import datasets
import numpy as np
import pandas as pd
import requests

"""XD-Violence dataset loader.

Adapted from:
- https://huggingface.co/datasets/jherng/xd-violence/blob/main/xd-violence.py
Dataset card:
- https://huggingface.co/datasets/jherng/xd-violence

License: MIT (original).
"""

ORIGINAL_SOURCE_URL = (
    "https://huggingface.co/datasets/jherng/xd-violence/blob/main/xd-violence.py"
)
DATASET_CARD_URL = "https://huggingface.co/datasets/jherng/xd-violence"


_CITATION = """\
@inproceedings{Wu2020not,
    title={Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision},
    author={Wu, Peng and Liu, jing and Shi, Yujia and Sun, Yujia and Shao, Fangtao and Wu, Zhaoyang and Yang, Zhiwei},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2020}
}
"""

_DESCRIPTION = """\
Dataset for the paper "Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision". \
The dataset is downloaded from the authors' website (https://roc-ng.github.io/XD-Violence/). Hosting this dataset on HuggingFace \
is just to make it easier for my own project to use this dataset. Please cite the original paper if you use this dataset.
"""

_NAME = "xd-violence"

_HOMEPAGE = f"https://huggingface.co/datasets/jherng/{_NAME}"

_LICENSE = "MIT"

_URL = f"https://huggingface.co/datasets/jherng/{_NAME}/resolve/main/data/"


class XDViolenceConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        """BuilderConfig for XD-Violence.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(XDViolenceConfig, self).__init__(**kwargs)


class XDViolence(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        XDViolenceConfig(
            name="video",
            description="Video dataset.",
        ),
        XDViolenceConfig(
            name="i3d_rgb",
            description="RGB features of the dataset extracted with pretrained I3D ResNet50 model (Pre-trained on ImageNet-1k; Transfer learning on Kinetics-400 dataset).",
        ),
        XDViolenceConfig(
            name="swin_rgb",
            description="RGB features of the dataset extracted with pretrained Video Swin Transformer model (Pre-trained on ImageNet-1k; Transfer learning on Kinetics-400 dataset).",
        ),
        XDViolenceConfig(
            name="c3d_rgb",
            description="RGB features of the dataset extracted with pretrained C3D model (Pre-trained on Sports-1M; Transfer learning on UCF-101 dataset).",
        ),
    ]

    DEFAULT_CONFIG_NAME = "video"
    BUILDER_CONFIG_CLASS = XDViolenceConfig

    CODE2IDX = {
        "A": 0,  # Normal
        "B1": 1,  # Fighting
        "B2": 2,  # Shooting
        "B4": 3,  # Riot
        "B5": 4,  # Abuse
        "B6": 5,  # Car accident
        "G": 6,  # Explosion
    }

    def _info(self):
        if self.config.name == "i3d_rgb":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "feature": datasets.Array2D(
                        shape=(None, 2048), dtype="float32"
                    ),  # (num_frames, feature_dim)
                    "binary_target": datasets.ClassLabel(
                        names=["Non-violence", "Violence"]
                    ),
                    "multilabel_target": datasets.Sequence(
                        datasets.ClassLabel(
                            names=[
                                "Normal",
                                "Fighting",
                                "Shooting",
                                "Riot",
                                "Abuse",
                                "Car accident",
                                "Explosion",
                            ]
                        )
                    ),
                    "frame_annotations": datasets.Sequence(
                        {
                            "start": datasets.Value("int32"),
                            "end": datasets.Value("int32"),
                        }
                    ),
                }
            )
        elif self.config.name == "swin_rgb":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "feature": datasets.Array2D(
                        shape=(None, 768), dtype="float32"
                    ),  # (num_frames, feature_dim)
                    "binary_target": datasets.ClassLabel(
                        names=["Non-violence", "Violence"]
                    ),
                    "multilabel_target": datasets.Sequence(
                        datasets.ClassLabel(
                            names=[
                                "Normal",
                                "Fighting",
                                "Shooting",
                                "Riot",
                                "Abuse",
                                "Car accident",
                                "Explosion",
                            ]
                        )
                    ),
                    "frame_annotations": datasets.Sequence(
                        {
                            "start": datasets.Value("int32"),
                            "end": datasets.Value("int32"),
                        }
                    ),
                }
            )
        elif self.config.name == "c3d_rgb":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "feature": datasets.Array2D(
                        shape=(None, 4096), dtype="float32"
                    ),  # (num_frames, feature_dim)
                    "binary_target": datasets.ClassLabel(
                        names=["Non-violence", "Violence"]
                    ),
                    "multilabel_target": datasets.Sequence(
                        datasets.ClassLabel(
                            names=[
                                "Normal",
                                "Fighting",
                                "Shooting",
                                "Riot",
                                "Abuse",
                                "Car accident",
                                "Explosion",
                            ]
                        )
                    ),
                    "frame_annotations": datasets.Sequence(
                        {
                            "start": datasets.Value("int32"),
                            "end": datasets.Value("int32"),
                        }
                    ),
                }
            )
        else:  # default = "video"
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "binary_target": datasets.ClassLabel(
                        names=["Non-violence", "Violence"]
                    ),
                    "multilabel_target": datasets.Sequence(
                        datasets.ClassLabel(
                            names=[
                                "Normal",
                                "Fighting",
                                "Shooting",
                                "Riot",
                                "Abuse",
                                "Car accident",
                                "Explosion",
                            ]
                        )
                    ),
                    "frame_annotations": datasets.Sequence(
                        {
                            "start": datasets.Value("int32"),
                            "end": datasets.Value("int32"),
                        }
                    ),
                }
            )

        return datasets.DatasetInfo(
            features=features,
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # Download train list
        train_list_path = dl_manager.download_and_extract(
            urllib.parse.urljoin(_URL, "train_list.txt")
        )
        print(train_list_path)
        train_list = (
            pd.read_csv(
                train_list_path, header=None, sep=" ", usecols=[0], names=["id"]
            )["id"]
            .apply(lambda x: x.rstrip(".mp4"))
            .tolist()
        )
        train_ids = [
            x.split("/")[1] for x in train_list
        ]  # remove subfolder prefix, e.g., "1-1004"

        # Download test list
        test_list_path = dl_manager.download_and_extract(
            urllib.parse.urljoin(_URL, "test_list.txt")
        )
        test_list = (
            pd.read_csv(
                test_list_path, header=None, sep=" ", usecols=[0], names=["id"]
            )["id"]
            .apply(lambda x: x.rstrip(".mp4"))
            .tolist()
        )
        test_ids = [x.split("/")[1] for x in test_list]

        # Download test annotation file
        test_annotations_path = dl_manager.download_and_extract(
            urllib.parse.urljoin(_URL, "test_annotations.txt")
        )

        if self.config.name == "i3d_rgb":
            # Download features
            train_paths = dl_manager.download(
                [
                    urllib.parse.quote(
                        urllib.parse.urljoin(_URL, f"i3d_rgb/{x}.npy"), safe=":/"
                    )
                    for x in train_list
                ]
            )
            test_paths = dl_manager.download(
                [
                    urllib.parse.quote(
                        urllib.parse.urljoin(_URL, f"i3d_rgb/{x}.npy"), safe=":/"
                    )
                    for x in test_list
                ]
            )
        elif self.config.name == "swin_rgb":
            # Download features
            train_paths = dl_manager.download(
                [
                    urllib.parse.quote(
                        urllib.parse.urljoin(_URL, f"swin_rgb/{x}.npy"), safe=":/"
                    )
                    for x in train_list
                ]
            )
            test_paths = dl_manager.download(
                [
                    urllib.parse.quote(
                        urllib.parse.urljoin(_URL, f"swin_rgb/{x}.npy"), safe=":/"
                    )
                    for x in test_list
                ]
            )
        elif self.config.name == "c3d_rgb":
            # Download features
            train_paths = dl_manager.download(
                [
                    urllib.parse.quote(
                        urllib.parse.urljoin(_URL, f"c3d_rgb/{x}.npy"), safe=":/"
                    )
                    for x in train_list
                ]
            )
            test_paths = dl_manager.download(
                [
                    urllib.parse.quote(
                        urllib.parse.urljoin(_URL, f"c3d_rgb/{x}.npy"), safe=":/"
                    )
                    for x in test_list
                ]
            )
        else:
            # Download videos
            train_paths = dl_manager.download(
                [
                    urllib.parse.quote(
                        urllib.parse.urljoin(_URL, f"video/{x}.mp4"), safe=":/"
                    )
                    for x in train_list
                ]
            )
            # test_paths = dl_manager.download(
            #     [
            #         urllib.parse.quote(
            #             urllib.parse.urljoin(_URL, f"video/{x}.mp4"), safe=":/"
            #         )
            #         for x in test_list
            #     ]
            # )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "ids": train_ids,
                    "paths": train_paths,
                    "annotations_path": None,
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={
            #         "ids": test_ids,
            #         "paths": test_paths,
            #         "annotations_path": test_annotations_path,
            #     },
            # ),
        ]

    def _generate_examples(self, ids, paths, annotations_path):
        frame_annots_mapper = (
            self._read_frame_annotations(annotations_path)
            if annotations_path
            else dict()
        )
        labels = [self._extract_labels(f_id) for f_id in ids]  # Extract labels

        if (
            self.config.name == "i3d_rgb"
            or self.config.name == "swin_rgb"
            or self.config.name == "c3d_rgb"
        ):
            for key, (f_id, f_path, f_label) in enumerate(zip(ids, paths, labels)):
                binary, multilabel = f_label
                frame_annotations = frame_annots_mapper.get(f_id, [])
                feature = np.load(f_path)

                for crop_idx in range(feature.shape[1]):  # Loop over crops (5 crops)
                    yield (
                        f"{key}-{crop_idx}",
                        {
                            "id": f"{f_id}__{crop_idx}",
                            "feature": np.squeeze(feature[:, crop_idx, :]).reshape(
                                (-1, feature.shape[-1])
                            ),
                            "binary_target": binary,
                            "multilabel_target": multilabel,
                            "frame_annotations": frame_annotations,
                        },
                    )
        else:
            for key, (f_id, f_path, f_label) in enumerate(zip(ids, paths, labels)):
                binary, multilabel = f_label
                frame_annotations = frame_annots_mapper.get(f_id, [])

                yield (
                    key,
                    {
                        "id": f_id,
                        "path": f_path,
                        "binary_target": binary,
                        "multilabel_target": multilabel,
                        "frame_annotations": frame_annotations,
                    },
                )

    def _read_frame_annotations(self, path):
        mapper = {}
        is_url = urllib.parse.urlparse(path).scheme in ("http", "https")

        if is_url:
            with requests.get(path, stream=True) as r:
                r.raise_for_status()

                for line in r.iter_lines():
                    parts = line.decode("utf-8").strip().split(" ")
                    f_id = parts[0].rstrip(".mp4")
                    frame_annotations = [
                        {"start": parts[start_idx], "end": parts[start_idx + 1]}
                        for start_idx in range(1, len(parts), 2)
                    ]

                    mapper[f_id] = frame_annotations

        else:
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    f_id = parts[0].rstrip(".mp4")
                    frame_annotations = [
                        {"start": parts[start_idx], "end": parts[start_idx + 1]}
                        for start_idx in range(1, len(parts), 2)
                    ]

                    mapper[f_id] = frame_annotations

        return mapper

    def _extract_labels(self, f_id):
        """Extracts labels from a given file id."""
        codes = f_id.split("_")[-1].split("-")
        binary = 1 if len(codes) > 1 else 0
        multilabel = [self.CODE2IDX[code] for code in codes if code != "0"]
        return binary, multilabel


# --- Personal code below this line ---
def extract_video_links_and_labels_to_csv(
    dataset_builder, output_csv="video_links_labels.csv"
):
    """
    Extracts all train video download links and their labels from the XDViolence dataset
    and writes them to a CSV file.

    Args:
        dataset_builder (XDViolence): An instance of the XDViolence dataset builder.
        output_csv (str): Path to the output CSV file.
    """

    #  Read the train list file directly (no DownloadManager, no video downloads)
    train_list_url = urllib.parse.urljoin(_URL, "train_list.txt")
    r = requests.get(train_list_url, timeout=30)
    r.raise_for_status()

    rows = []
    for line in r.text.splitlines():
        parts = line.strip().split(" ")
        if not parts:
            continue
        item = parts[0].rstrip(".mp4")  # e.g. "<subfolder>/<id>"
        f_id = item.split("/")[1]  # keep the "<id>" (e.g., "1-1004")

        # 2) Build the public video URL exactly like the dataset does
        download_url = urllib.parse.quote(
            urllib.parse.urljoin(_URL, f"video/{item}.mp4"), safe=":/"
        )

        # 3) Reuse existing label extraction
        binary, multilabel = dataset_builder._extract_labels(f_id)

        rows.append(
            {
                "id": f_id,
                "download_url": download_url,
                "binary_label": binary,
                "multilabel": ";".join(str(x) for x in multilabel),
            }
        )

    # 4) Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["id", "download_url", "binary_label", "multilabel"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
