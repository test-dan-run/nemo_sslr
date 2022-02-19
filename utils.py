import os
import json
import pandas as pd
from pathlib import Path
from typing import List

# manifest_path should be located in the same directory as the audio files
# update manifest to update relative audio_filepaths to absolute paths
# returns new manifest path
def update_manifest_from_json(manifest_path: str) -> str:

    main_dir = os.path.dirname(manifest_path)
    new_manifest_path = os.path.join(
        main_dir, 
        'updated_' + os.path.basename(manifest_path)
        )
    with open(manifest_path, 'r') as fr, open(new_manifest_path, 'w') as fw:
        lines = fr.readlines()
        for line in lines:
            row = json.loads(line)
            row['audio_filepath'] = os.path.join(main_dir, row['audio_filepath'])
            fw.write(
                json.dumps(row) + '\n'
            )

    return new_manifest_path

def update_manifest_from_csv(csv_path: str, main_dir: str, manifest_path: str, additional_columns: List[str] = None) -> str:

    output_path = Path(os.path.join(main_dir, manifest_path))
    output_path.parent.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(csv_path)

    with output_path.open(mode='w', encoding='utf8') as fw:
        for _, row in df.iterrows():
            line = {
                'audio_filepath': os.path.join(main_dir, row['audio_filepath']),
                'duration': row['duration'],
                'text': row['text'].lower().strip(),
            }
            if additional_columns:
                for col in additional_columns:
                    line[col] = row[col]
            
            fw.write(json.dumps(line) + '\n')

    print('Generated manifest at:', str(output_path))
    return str(output_path)
