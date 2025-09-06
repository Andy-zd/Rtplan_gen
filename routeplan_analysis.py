import os
import json
import random

if __name__ == "__main__":
    for dataset in ['scannet', 'scannetpp', 'arkitscene']:
        json_file_path = f'./qa_pairs_{dataset}.json'

        # Load the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Find indices where turning_directions == ['Back']
        backs_idx = [idx for idx, d in enumerate(data) if d['turning_directions'] == ['Back']]

        # Randomly drop 75% of the 'Back' samples to balance the dataset
        invalid_idx = random.sample(backs_idx, len(backs_idx) * 3 // 4)
        data = [d for idx, d in enumerate(data) if idx not in invalid_idx]

        # ✅ Keep only samples where turning_directions length < 2
        data = [d for d in data if len(d['turning_directions']) < 2]

        # Count occurrences of different directions
        backs = [d for d in data if d['turning_directions'] == ['Back']]
        rights = [d for d in data if d['turning_directions'] == ['Right']]
        lefts = [d for d in data if d['turning_directions'] == ['Left']]

        print(f"Dataset: {dataset}")
        print(f"Number of 'Back' turning directions: {len(backs)}")
        print(f"Number of 'Right' turning directions: {len(rights)}")
        print(f"Number of 'Left' turning directions: {len(lefts)}")
        print("="*50)

        # Save the filtered data back to the JSON file
        with open(json_file_path, 'w') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)