import datetime
import json
import os

from google.cloud import storage


def extract_vision_data(json_path: str) -> dict:
    """
    Build a mapping from each vision label to its associated_objects list.

    Args:
        vision_data (dict): The 'vision' section of the taxonomy JSON
                            (i.e. data['vision'] after loading labels.json).

    Returns:
        dict[str, list[str]]: { label_id -> associated_objects }

    Notes:
        - Violent labels inherit the associated_objects defined at their parent category.
        - Non-violent and neutral labels carry their own associated_objects directly.
    """
    with open(json_path, "r") as file:
        vision_data = json.load(file)

    vision_data = vision_data["vision"]

    vision_labels = {}
    associated_objects = set()
    # Violent categories: objects live at category level
    for category in vision_data["violent"]:
        cat_objs = category["associated_objects"]
        for label in category["labels"]:
            label_name = label["description"]
            vision_labels[label_name] = {
                "associated_objects": list(cat_objs),
                "category": "violent",
            }
            associated_objects.update(cat_objs)

    # Non-violent & neutral: objects are on each label entry
    for section_key in ("non_violent", "neutral"):
        for label in vision_data[section_key]:
            vision_labels[label["description"]] = {
                "associated_objects": list(label["associated_objects"]),
                "category": section_key,
            }
            associated_objects.update(cat_objs)

    return vision_labels, list(associated_objects)


def extract_audio_data(json_path: str) -> dict:
    """
    Build a mapping from each audio label to its associated_audioset_labels list,
    and also return a list of all unique associated audioset labels.

    Args:
        json_path (str): Path to the taxonomy JSON (labels.json).

    Returns:
        tuple:
            audio_labels_dict (dict): {label_description: {"associated_audioset_labels": [...], "category": ...}}
            unique_audioset_labels (list): List of all unique audioset label names.
    """
    with open(json_path, "r") as file:
        audio_data = json.load(file)

    audio_data = audio_data["audio"]

    audio_labels = {}
    unique_audioset_labels = set()

    # Violent categories: objects live at category level
    for category in audio_data["violent"]:
        for label in category["labels"]:
            label_name = label["description"]
            audioset_labels = [
                x["name"] for x in label.get("associated_audioset_labels", [])
            ]
            audio_labels[label_name] = {
                "associated_audioset_labels": audioset_labels,
                "category": "violent",
            }
            unique_audioset_labels.update(audioset_labels)

    # Non-violent & neutral: audioset labels are on each label entry
    for section_key in ("non_violent", "neutral"):
        for label in audio_data[section_key]:
            label_name = label["description"]
            audioset_labels = [
                x["name"] for x in label.get("associated_audioset_labels", [])
            ]
            audio_labels[label_name] = {
                "associated_audioset_labels": audioset_labels,
                "category": section_key,
            }
            unique_audioset_labels.update(audioset_labels)

    return audio_labels, sorted(unique_audioset_labels)


def list_all_blobs():
    """
    Return all object names in the bucket.
     Returns:
        list[str]: Full object names for every blob in the bucket.
    """
    bucket_name = os.environ["BUCKET_NAME"]
    client = storage.Client(project="multimodal-violence-detection")
    return [b.name for b in client.list_blobs(bucket_name)]


def list_all_blobs_in_subfolder(folder_name):
    """
    Return all object names that are under a given "folder".
    Args:
        folder_name (str): Prefix representing the subfolder to search.

    Returns:
        list[str]: Object names under the specified prefix.
    """
    bucket_name = os.environ["BUCKET_NAME"]
    client = storage.Client(project="multimodal-violence-detection")
    return [
        b.name
        for b in client.list_blobs(bucket_name)
        if b.name.startswith(folder_name + "/")
    ]


def generate_download_signed_url_v4(blob_name):
    """Generates a v4 signed URL for downloading a blob.
    Args:
        blob_name (str): Full object name (path) in the bucket.

    Returns:
        str: A signed URL valid for 15 minutes.

    Note that this method requires a service account key file.

    Requirements:
        - The BUCKET_NAME environment variable must be set.
        - Credentials must be able to sign URLs, either:
          • A service account key file used by the client, or
          • An identity with iam.serviceAccounts.signBlob permission for the
            signing service account (via IAM or impersonation).
    """
    bucket_name = os.environ["BUCKET_NAME"]

    storage_client = storage.Client(project="multimodal-violence-detection")
    print(storage_client.bucket())
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        # This URL is valid for 15 minutes
        expiration=datetime.timedelta(minutes=15),
        # Allow GET requests using this URL.
        method="GET",
    )
    return url
