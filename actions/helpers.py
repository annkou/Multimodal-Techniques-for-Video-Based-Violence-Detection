import datetime
import json
import os
import pandas as pd 
from google.cloud import storage
import copy
from collections import defaultdict


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


def check_model_availability(video_paths: list[str], model_files:dict):
    """
    Check which models have results for each video.
    
    Args:
        video_paths: List of video file paths
        model_files: Dict of model_name -> results.json path
    Returns:
        DataFrame with video_path and model availability columns
    """
    # Initialize with video info
    availability_list = []
    for video_path, info in video_paths.items():
        availability_list.append({
            'video_name': info['video_name'],
            'local_path': video_path,
            'download_url': info['download_url']
        })

    availability = pd.DataFrame(availability_list)
    # Initialize all model columns with False
    for model_name in model_files.keys():
        availability[model_name] = False
        
    # Check each model
    for model_name, file_path in model_files.items():
        print(f"Checking {model_name}...")

        with open(file_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Get set of available video paths/names
        # Handle both URLs and local paths
        available_videos = set()
        for r in results:
            if 'video_path' not in r or 'error' in r:
                continue
            
            video_path_result = r.get('video_path', '')
            
            # If it's a URL, keep it whole
            if video_path_result.startswith('https://') or video_path_result.startswith('http://'):
                available_videos.add(video_path_result)
            else:
                # For local paths, extract basename without .mp4
                basename = os.path.basename(video_path_result).split('.mp4')[0]
                available_videos.add(basename)
        print(f"  Found {len(available_videos)} videos with results")

        # Update availability for each video
        for idx, row in availability.iterrows():
            video_name = row['video_name']
            url = row['download_url']
            
            # Check if video_name or url is in available_videos
            if video_name in list(available_videos) or url in list(available_videos):
                # print(f"  Video available: {video_name} or {url}")
                availability.at[idx, model_name] = True
        
    for model_name in model_files.keys():
        count = availability.loc[availability[model_name]==True].shape[0]
        percentage = (count / len(availability) * 100) if len(availability) > 0 else 0
        print(f"  {model_name:20s}: {count:4d} ({percentage:5.1f}%)")
    print(f"{'='*60}\n")
    return availability
            

def filter_and_prepare_model_results(
    video_paths_df: pd.DataFrame,
    model_files: dict,
    probability_threshold: float = 0.1
):
    """
    Extract and filter model results for each video.
    For vision/audio models (clip, xclip, clap, beats), filter out labels with probability < threshold.
    
    Args:
        video_paths_df: Availability dataframe with model availability columns 
        model_files: Dict of model_name -> results.json path
        probability_threshold: Minimum probability to keep labels (default: 0.1)
    
    Returns:
        Dict mapping video_name -> {model_name: filtered_response}
    """
    # Models that need probability filtering
    filter_models = {'clip', 'xclip', 'clap', 'beats'}
    
    # Store filtered results
    video_model_results = {}

    
    # Initialize structure
    for video_name in video_paths_df['video_name']:
        video_model_results[video_name] = {}
    
    # Process each model
    for model_name, file_path in model_files.items():
        print(f"\nProcessing {model_name}...")
        
        
        with open(file_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        # Process each video's results
        for result in results:
            # print(result)
            video_name = result.get('video_path', '')
            
            # Check if video exists in dataset by video_name or download_url
            video_match = video_paths_df[
                (video_paths_df['video_name'] == os.path.basename(video_name).split('.mp4')[0]) | 
                (video_paths_df['download_url'] == video_name)
            ]
            # Skip if video not in dataset
            if video_match.empty:
                continue

            # Get the matched video name
            matched_video_name = video_match.iloc[0]['video_name']
            # print(f"  Processing video: {matched_video_name}")
            
            # Check if this model has results for this video (availability check)
            if model_name in video_paths_df.columns:
                has_model = video_match.iloc[0][model_name]
                if not has_model:  # Skip if False
                    continue

            modality = result['modality']

            response = result.get('response', [])
            
            # Apply filtering for specific models
            if model_name in filter_models and isinstance(response, list):
                filtered_response = []
                for item in response:
                    if 'labels' in item and isinstance(item['labels'], dict):
                        # Filter labels by probability threshold
                        filtered_labels = {
                            label: prob
                            for label, prob in item['labels'].items()
                            if prob >= probability_threshold
                        }
                        
                        # Only keep item if at least one label remains
                        if filtered_labels:
                            filtered_item = item.copy()
                            filtered_item['labels'] = filtered_labels
                            filtered_response.append(filtered_item)
                    else:
                        # Keep items without labels dict as-is
                        filtered_response.append(item)
                
                # Append to existing results if model already has data for this video
                if model_name in video_model_results[matched_video_name]:
                    video_model_results[matched_video_name][model_name]['response'].extend(filtered_response)
                else:
                    video_model_results[matched_video_name][model_name] = {
                        'response': filtered_response,
                        'modality': modality
                    }
                
            else:
                # No filtering needed for other models (whisper, gemini, qwen)
                # For these models, replace (don't append) since they shouldn't have duplicates
                video_model_results[matched_video_name][model_name] = {
                    'response': response,
                    'modality': modality
                }
        
        print(f"  ✅ Processed {len(results)} videos")
        
    
    # Print statistics
    print(f"\n{'='*60}")
    print("FILTERING STATISTICS")
    print(f"{'='*60}")
    
    for model_name in filter_models:
        if model_name in model_files:
            total_items = 0
            filtered_items = 0
            videos_with_results = 0
            
            for video_name, models in video_model_results.items():
                if model_name in models:
                    videos_with_results += 1
                    response = models[model_name]
                    if isinstance(response, list):
                        filtered_items += len(response)
            
            print(f"\n{model_name}:")
            print(f"  Videos with results: {videos_with_results}")
    
    print(f"{'='*60}\n")
    
    return video_model_results


def parse_batch_results_to_json(file_response, output_json: str):
    """
    Parse batch API response and save to JSON file.
    
    Args:
        file_response: Response from download_batch_output
        output_json: Path to output JSON file
    
    Returns:
        list: Parsed results
    """
    response_lines = file_response.content.decode('utf-8').strip().split('\n')
    
    parsed_results = []
    
    for line in response_lines:
        result = json.loads(line)
        
        # Extract custom_id and video name
        custom_id = result.get('custom_id', 'unknown')
        video_name = custom_id.split('video-')[-1].split('-', 1)[-1] if 'video-' in custom_id else custom_id
        
        # Extract response body
        response_body = result.get('response', {}).get('body', {})
        choices = response_body.get('choices', [])
        
        if choices:
            content = choices[0].get('message', {}).get('content', '')
            evaluation = json.loads(content)
            
            # Get usage stats
            usage = response_body.get('usage', {})
            
            # Build structured result
            parsed_result = {
                "video_name": video_name,
                "custom_id": custom_id,
                "violence_probability": evaluation.get('violence_probability'),
                "confidence": evaluation.get('confidence'),
                "abstain": evaluation.get('abstain'),
                "rationale": evaluation.get('rationale'),
                "primary_modalities": evaluation.get('primary_modalities', []),
                "primary_models": evaluation.get('primary_models', []),
                "usage": {
                    "prompt_tokens": usage.get('prompt_tokens', 0),
                    "completion_tokens": usage.get('completion_tokens', 0),
                    "total_tokens": usage.get('total_tokens', 0),
                    "reasoning_tokens": usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)
                },
                "model": response_body.get('model'),
                "request_id": result.get('response', {}).get('request_id')
            }
            
            parsed_results.append(parsed_result)
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(parsed_results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(parsed_results)} results to: {output_json}")
    
    return parsed_results


def parse_batch_results_to_csv(file_response, output_csv: str):
    """
    Parse batch API response and save to CSV file.
    
    Args:
        file_response: Response from download_batch_output
        output_csv: Path to output CSV file
    
    Returns:
        DataFrame: Parsed results
    """
    response_lines = file_response.content.decode('utf-8').strip().split('\n')
    
    rows = []
    
    for line in response_lines:
        result = json.loads(line)
        
        # Extract custom_id and video name
        custom_id = result.get('custom_id', 'unknown')
        video_name = custom_id.split('video-')[-1].split('-', 1)[-1] if 'video-' in custom_id else custom_id
        
        # Extract response body
        response_body = result.get('response', {}).get('body', {})
        choices = response_body.get('choices', [])
        
        if choices:
            content = choices[0].get('message', {}).get('content', '')
            evaluation = json.loads(content)
            
            # Get usage stats
            usage = response_body.get('usage', {})
            
            # Build row
            row = {
                "video_name": video_name,
                "custom_id": custom_id,
                "violence_probability": evaluation.get('violence_probability'),
                "confidence": evaluation.get('confidence'),
                "abstain": evaluation.get('abstain'),
                "rationale": evaluation.get('rationale'),
                "primary_modalities": '|'.join(evaluation.get('primary_modalities', [])),
                "primary_models": '|'.join(evaluation.get('primary_models', [])),
                "prompt_tokens": usage.get('prompt_tokens', 0),
                "completion_tokens": usage.get('completion_tokens', 0),
                "total_tokens": usage.get('total_tokens', 0),
                "reasoning_tokens": usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0),
                "model": response_body.get('model'),
                "request_id": result.get('response', {}).get('request_id')
            }
            
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"Saved {len(df)} results to: {output_csv}")
    
    return df

def merge_consecutive_segments(
    filtered_results: dict,
    models_to_merge: list = ['clip', 'xclip'],
    time_tolerance: float = 0.0
):
    """
    Merge consecutive segments in CLIP/XCLIP results that have the same labels.
    
    Args:
        filtered_results: Dict mapping video_name -> {model_name: {'response': [...], 'modality': '...'}}
        models_to_merge: List of model names to apply merging (default: ['clip', 'xclip'])
        time_tolerance: Maximum allowed gap between segments to merge (default: 0.0)
    
    Returns:
        dict: New dict with merged results
    """
    
    # Create a deep copy to avoid modifying original
    merged_results = copy.deepcopy(filtered_results)
    
    # Track statistics
    stats = defaultdict(lambda: {'original': 0, 'merged': 0, 'videos_processed': 0})
    
    # Process each video
    for video_name, models_data in merged_results.items():
        # Process each model that needs merging
        for model_name in models_to_merge:
            if model_name not in models_data:
                continue
            
            response = models_data[model_name].get('response', [])
            if not isinstance(response, list) or len(response) == 0:
                continue
            
            original_count = len(response)
            stats[model_name]['original'] += original_count
            
            # Sort by start_time to ensure chronological order
            # response.sort(key=lambda x: x.get('start_time', 0))
            
            merged_segments = []
            current_segment = None
            
            for segment in response:
                # Extract segment data
                start_time = segment.get('start_time', 0)
                end_time = segment.get('end_time', 0)
                labels = segment.get('labels', {})
                
                # Skip segments without labels
                if not labels:
                    if current_segment:
                        merged_segments.append(current_segment)
                        current_segment = None
                    merged_segments.append(segment)
                    continue
                
                # Get label keys as a set for comparison
                current_label_keys = set(labels.keys())
                
                # If this is the first segment, initialize current_segment
                if current_segment is None:
                    current_segment = {
                        'start_time': start_time,
                        'end_time': end_time,
                        'labels': labels.copy(),
                        'segment_count': 1,
                        'label_sums': labels.copy()  # Track sums for averaging
                    }
                    continue
                
                # Check if we can merge with current_segment
                prev_end_time = current_segment['end_time']
                prev_label_keys = set(current_segment['labels'].keys())
                
                # Conditions for merging:
                # 1. Same label keys
                # 2. Previous end_time <= current start_time (with tolerance)
                can_merge = (
                    prev_label_keys == current_label_keys and
                    prev_end_time <= start_time + time_tolerance
                )
                
                if can_merge:
                    # Merge: extend end_time and accumulate label probabilities
                    current_segment['end_time'] = end_time
                    current_segment['segment_count'] += 1
                    
                    # Add probabilities to sums
                    for label, prob in labels.items():
                        current_segment['label_sums'][label] += prob
                    
                else:
                    # Can't merge - finalize current segment and start new one
                    # Calculate mean probabilities
                    count = current_segment['segment_count']
                    averaged_labels = {
                        label: round(sum_prob / count, 3)
                        for label, sum_prob in current_segment['label_sums'].items()
                    }
                    
                    # Sort by probability (descending)
                    averaged_labels = dict(sorted(
                        averaged_labels.items(),
                        key=lambda x: x[1],
                        reverse=True
                    ))
                    
                    # Create final merged segment
                    final_segment = {
                        'start_time': current_segment['start_time'],
                        'end_time': current_segment['end_time'],
                        'labels': averaged_labels
                    }
                    merged_segments.append(final_segment)
                    
                    # Start new segment
                    current_segment = {
                        'start_time': start_time,
                        'end_time': end_time,
                        'labels': labels.copy(),
                        'segment_count': 1,
                        'label_sums': labels.copy()
                    }
            
            # Don't forget to add the last segment
            if current_segment is not None:
                count = current_segment['segment_count']
                averaged_labels = {
                    label: sum_prob / count
                    for label, sum_prob in current_segment['label_sums'].items()
                }
                
                # Sort by probability (descending)
                averaged_labels = dict(sorted(
                    averaged_labels.items(),
                    key=lambda x: x[1],
                    reverse=True
                ))
                
                final_segment = {
                    'start_time': current_segment['start_time'],
                    'end_time': current_segment['end_time'],
                    'labels': averaged_labels
                }
                merged_segments.append(final_segment)
            
            # Update the response with merged segments
            merged_results[video_name][model_name]['response'] = merged_segments
            
            merged_count = len(merged_segments)
            stats[model_name]['merged'] += merged_count
            stats[model_name]['videos_processed'] += 1
            
            # Log reduction for this video if significant
            reduction = original_count - merged_count
            if reduction > 0:
                reduction_pct = (reduction / original_count) * 100
                if reduction_pct > 20:  # Only log if >20% reduction
                    print(f"  {video_name[:50]:50s} | {model_name:5s}: {original_count:3d} → {merged_count:3d} ({reduction_pct:.1f}% reduction)")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SEGMENT MERGING STATISTICS")
    print(f"{'='*80}")
    
    for model_name in models_to_merge:
        if model_name in stats:
            s = stats[model_name]
            reduction = s['original'] - s['merged']
            reduction_pct = (reduction / s['original'] * 100) if s['original'] > 0 else 0
            
            print(f"\n{model_name.upper()}:")
            print(f"  Videos processed: {s['videos_processed']}")
            print(f"  Original segments: {s['original']:,}")
            print(f"  Merged segments: {s['merged']:,}")
            print(f"  Reduction: {reduction:,} segments ({reduction_pct:.1f}%)")
    
    print(f"{'='*80}\n")
    
    return merged_results

def append_results(results_paths):
    dfs = [pd.read_csv(new_results_path) for new_results_path in results_paths]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df