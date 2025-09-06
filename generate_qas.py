import numpy as np
import os
import json
def unique_dicts(dicts):
    """
    Remove duplicates from a list of dictionaries using JSON serialization.
    """
    unique_json = set(json.dumps(d, sort_keys=True) for d in dicts)
    return [json.loads(u) for u in unique_json]

def query_obj(unaligned_bbox, pt, object_lists):
    x_center, y_center, width, height = unaligned_bbox[:, 0], unaligned_bbox[:, 1], unaligned_bbox[:, 3], unaligned_bbox[:, 4]

    corners_x = np.vstack((x_center - width / 2, x_center + width / 2, x_center - width / 2, x_center + width / 2)).T
    corners_y = np.vstack((y_center - height / 2, y_center - height / 2, y_center + height / 2, y_center + height / 2)).T

    pt_x, pt_y = np.tile(pt[0], (unaligned_bbox.shape[0], 4)), np.tile(pt[1], (unaligned_bbox.shape[0], 4))

    distances = np.sqrt((corners_x - pt_x) ** 2 + (corners_y - pt_y) ** 2)

    min_distances = np.min(distances, axis=1)

    id = np.argmin(min_distances)
    
    return object_lists[id], min_distances[id],id

def find_all_indices_for_duplicates(object_list):
    # Create a dictionary to record the number of occurrences of each object
    counts = {}
    # Create a dictionary to record the first index where each object appears
    first_indices = {}
    # Create a list to collect all indices of duplicate objects
    duplicate_indices = []
    
    for idx, obj in enumerate(object_list):
        if obj in counts:
            # Increase the occurrence count for the object
            counts[obj] += 1
            # When the object appears for the 2nd time, add its first occurrence index to the list
            if counts[obj] == 2:
                duplicate_indices.append(first_indices[obj])
            # Add the current index to the list
            duplicate_indices.append(idx)
        else:
            # Initialize the occurrence count and the first index of the object
            counts[obj] = 1
            first_indices[obj] = idx
            
    return duplicate_indices

# serveral templates for prompt generation

# Template 1: no turning points
def Template_1(SRC, MID, TGT):
    starts = f'You are a robot beginning at the {MID} facing the {TGT}. You want to navigate to the {SRC}. '
    actions = "You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'):  "

    procedures = f"1. [please fill in] "
    procedures = procedures + f"2. Go forward until the {SRC}. "
    ends = "You have reached the final destination."
    

    final_prompt = starts + actions + procedures + ends
    turning_directions = np.array(['Back'])
    return final_prompt, turning_directions

# Template 2: multiple turning points
def Template_2(SRC, FACEs, TGT,turning_directions):
    starts = f'You are a robot beginning at the {SRC} facing the {FACEs[0]}. You want to navigate to the {TGT}. '
    actions = "You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'):  "
    for idx,FACE in enumerate(FACEs):
        procedures = f"{2*idx+1}. Go forward until the {FACE} {2*idx+2}. [please fill in] "
    procedures = procedures + f"{2*(idx+1)+1}. Go forward until the {TGT}. "
    ends = "You have reached the final destination."
    
    final_prompt = starts + actions + procedures + ends

    return final_prompt, turning_directions

# Template 3: only one turning point
def Template_3(SRC, FACE, TGT,turning_directions):

    starts = f'You are a robot beginning at the {FACE} facing the {TGT}. You want to navigate to the {SRC}. '
    actions = "You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'):  "

    procedures = f"1. [please fill in] "
    procedures = procedures + f"2. Go forward until the {SRC}. "
    ends = "You have reached the final destination."

    final_prompt = starts + actions + procedures + ends
    # turning_directions = np.array(['Back'])
    return final_prompt, turning_directions

def Template_4(SRC, FACEs, TGT,turning_direction):
    assert len(FACEs) == 2
    starts = f'You are a robot beginning at the {SRC} facing the {FACEs[0]}. You want to navigate to the {TGT}. '
    actions = "You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'):  "

    
    procedures = f"1. [please fill in] 2.  Go forward until the {FACEs[1]}.  "
    procedures = procedures + f"3. [please fill in] 4. Go forward until the {TGT}. "
    ends = "You have reached the final destination."

    final_prompt = starts + actions + procedures + ends
    turning_directions = np.array(['Back', turning_direction.item()])
    return final_prompt, turning_directions


def get_obj_list_scannetpp(data,scene_name):
    # object_bboxes = data[scene_name]['object_bboxes']
    object_names = []
    object_bboxes = []

    # Iterate through all object types in the scene
    for obj_type in data[scene_name]['object_bboxes'].keys():
        # Get all instances of this object type
        instances = data[scene_name]['object_bboxes'][obj_type]
        
        # For each instance, add its name and bbox info to the lists
        for instance in instances:
            # Get the bbox data for this instance
            bbox_data = instances[instance] if isinstance(instances, dict) else instance
            
            # Add the object name
            object_names.append(obj_type)
            
            # Create the 6D vector: centroid coordinates + axes lengths
            bbox_vector = [
                bbox_data['centroid'][0],  # x
                bbox_data['centroid'][1],  # y
                bbox_data['centroid'][2],  # z
                bbox_data['axesLengths'][0],  # length x
                bbox_data['axesLengths'][1],  # length y
                bbox_data['axesLengths'][2]   # length z
            ]
            object_bboxes.append(bbox_vector)

    return object_names, object_bboxes



def generate_rt_plan(scene_name, 
                     generated_path=None, 
                     data=None,
                     skip_repeated=True, 
                     skip_repeated_items=False,
                     return_other_info=False,
                     DEGREE_THRESHOLD1=30, 
                     DEGREE_THRESHOLD2=45):
    """
    Unified function for scannet, scannetpp and arkitscene.
    """

    qa_paris = []

    # ----------------------
    # Step 1: Load objects & bbox
    # ----------------------
    
    if scene_name not in data or scene_name not in os.listdir(generated_path):
        return []

    object_lists, object_bboxes = get_obj_list_scannetpp(data, scene_name)
    unaligned_bbox = np.array(object_bboxes)

    src_root = generated_path

   

    # ----------------------
    # Step 2: Clean object list
    # ----------------------
    invalid_labels = ['wall', 'object', 'floor', 'ceiling']
    valid_idx = np.array([i for i, label in enumerate(object_lists) if label not in invalid_labels])

    if skip_repeated_items:
        repeated_items = find_all_indices_for_duplicates(object_lists)
        valid_idx = np.array([i for i in valid_idx if i not in repeated_items])

    if len(valid_idx) == 0:
        print(f'No valid objects found in {scene_name}')
        return []

    object_lists = np.array(object_lists)[valid_idx]
    unaligned_bbox = unaligned_bbox[valid_idx]

    # ----------------------
    # Step 3: Iterate seeds/poses
    # ----------------------
    seeds = os.listdir(f'{src_root}/{scene_name}')
    for seed in seeds:
        poses_file = f'{src_root}/{scene_name}/{seed}/poses.txt'
        try:
            poses = np.loadtxt(poses_file)
        except:
            continue

        # fix coordinate system
        poses[:, 2] = -poses[:, 2]

        poses_list = [poses, poses[::-1]]
        if len(poses) > 8:
            poses_list += [
                poses[:len(poses) * 3 // 4],
                poses[:len(poses) * 3 // 4][::-1],
                poses[len(poses) // 4:],
                poses[len(poses) // 4:][::-1]
            ]

        for poses_ in poses_list:
            pos_idx = [0, 2]
            start_point = poses_[0, pos_idx]
            end_point = poses_[-1, pos_idx]

            differences = np.diff(poses_[:, pos_idx], axis=0)
            angles = np.array([
                np.arccos(np.clip(np.dot(differences[i], differences[i + 1]) /
                                  (np.linalg.norm(differences[i]) * np.linalg.norm(differences[i + 1])), -1.0, 1.0))
                for i in range(len(differences) - 1)
            ])
            angles_deg = np.degrees(angles)

            threshold = DEGREE_THRESHOLD1
            turning_points_idx = np.where(angles_deg > threshold)[0] + 1
            turning_degrees = angles_deg[turning_points_idx - 1]

            SRC, _, src_id = query_obj(unaligned_bbox, start_point, object_lists)
            TGT, _, tgt_id = query_obj(unaligned_bbox, end_point, object_lists)

            if len(turning_points_idx) == 0:
                mid_points = poses_[np.ix_([len(poses_) // 2], pos_idx)]
                MID, _, mid_id = query_obj(unaligned_bbox, mid_points[0], object_lists)

                sequences = [SRC, MID, TGT]
                sequences_id = [src_id, mid_id, tgt_id]

                if skip_repeated and len(sequences) != len(set(sequences)):
                    continue

                final_prompt, turning_directions = Template_1(SRC, MID, TGT)
                final_prompt, turning_directions = [final_prompt], [turning_directions]
                sequences_ids = [sequences_id]

            else:
                cross_products = np.cross(differences[:-1], differences[1:])
                turn_directions = np.where(cross_products > 0, 'Left', 'Right')
                turning_directions = turn_directions[turning_points_idx - 1]

                mid_points = poses_[np.ix_(turning_points_idx, pos_idx)]
                FACEs, FACEs_id = [], []
                for mid_point in mid_points:
                    FACE, _, id_ = query_obj(unaligned_bbox, mid_point, object_lists)
                    FACEs.append(FACE)
                    FACEs_id.append(id_)

                sequences = [SRC] + FACEs + [TGT]
                sequences_id = [src_id] + FACEs_id + [tgt_id]

                if skip_repeated and len(sequences) != len(set(sequences)):
                    continue

                final_prompt, turning_directions = Template_2(SRC, FACEs, TGT, turning_directions)
                final_prompt, turning_directions = [final_prompt], [turning_directions]
                sequences_ids = [sequences_id]

                if len(turning_directions[0]) == 1 and len(FACEs) == 1:
                    if turning_degrees[0] > DEGREE_THRESHOLD2:
                        prompt, directions = Template_3(SRC, FACEs[0], TGT, turning_directions[0])
                        final_prompt.append(prompt)
                        turning_directions.append(directions)
                        sequences_ids.append([sequences_id])
                    else:
                        print(f'The turning degree is less than {DEGREE_THRESHOLD2}, current={turning_degrees[0]}')

            for idx in range(len(final_prompt)):
                qa_paris.append({
                    'scene_name': scene_name,
                    'final_prompt': final_prompt[idx],
                    'turning_directions': turning_directions[idx].tolist(),
                    'seed': seed,
                    'bboxes': [unaligned_bbox[id_].tolist() for id_ in sequences_ids[idx]]
                })

    return qa_paris
    
if __name__ == "__main__":

    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_data_path', type=str, required=True, help='path to the processed data')
    parser.add_argument('--dataset_type', type=str, required=True, help='dataset type: scannet, scannetpp, arkitscenes')
    args = parser.parse_args()
    processed_data_path = args.processed_data_path
    dataset_type = args.dataset_type # 'scannet', 'scannetpp', 'arkitscenes'

    # v1
    DEGREE_THRESHOLD1 = 30
    DEGREE_THRESHOLD2 = 45


    skip_repeated_items = True
    return_other_info = True
    
    
    import os, json
    def unique_dicts(dicts):
        """
        Remove duplicates from a list of dictionaries using JSON serialization.
        """
        unique_json = set(json.dumps(d, sort_keys=True) for d in dicts)
        return [json.loads(u) for u in unique_json]


    def process_dataset(scenes, generator, **gen_kwargs):
        QA_paris, unavailables = [], []
        for scene_name in scenes:
            try:
                qa_paris = generator(scene_name, **gen_kwargs)
                qa_paris_unique = unique_dicts(qa_paris)
                QA_paris.extend(qa_paris_unique)
            except FileNotFoundError as e:
                print(f"[Error] {scene_name} is not available: {e}")
                unavailables.append(scene_name)
                continue
        return QA_paris, unavailables


    def main(
            dataset_path,
            dataset_type,
            skip_repeated_items=False,
            return_other_info=False,
            DEGREE_THRESHOLD1=30,
            DEGREE_THRESHOLD2=45):

        QA_paris, unavailables = [], []

        if dataset_type == "scannet":
            scenes = sorted(os.listdir(
                f"{dataset_path}/scans"
            ))
            with open(
                f"{processed_data_path}/ScanNet/metadata/train/scannetpp_metadata_train.json",
                "r",
            ) as f:
                metadata = json.load(f)
            QA_paris, unavailables = process_dataset(
                scenes,
                generator=lambda scene_name, **_: generate_rt_plan(
                    scene_name,
                    generated_path=generated_path,
                    data=metadata,
                    skip_repeated=True,
                    skip_repeated_items=skip_repeated_items,
                    return_other_info=return_other_info,
                    DEGREE_THRESHOLD1=DEGREE_THRESHOLD1,
                    DEGREE_THRESHOLD2=DEGREE_THRESHOLD2,
                )
            )

            # Example: save only for scannet, others可选
            with open("qa_pairs_repeated.json", "w", encoding="utf-8") as f:
                json.dump(QA_paris, f, ensure_ascii=False, indent=4)

        elif dataset_type == "scannetpp":
            scenes = sorted(os.listdir(f"{dataset_path}/data"))
            with open(
                f"{processed_data_path}/ScanNetpp/metadata/train/scannetpp_metadata_train.json",
                "r",
            ) as f:
                metadata = json.load(f)

            generated_path = "./generated_paths/scannetpp"
            QA_paris, unavailables = process_dataset(
                scenes,
                generator=lambda scene_name, **_: generate_rt_plan(
                    scene_name,
                    generated_path=generated_path,
                    data=metadata,
                    skip_repeated=True,
                    skip_repeated_items=skip_repeated_items,
                    return_other_info=return_other_info,
                    DEGREE_THRESHOLD1=DEGREE_THRESHOLD1,
                    DEGREE_THRESHOLD2=DEGREE_THRESHOLD2,
                )
            )

        elif dataset_type == "arkitscenes":
            scenes = os.listdir(f"{dataset_path}/raw/Training")
            with open(
                f"{processed_data_path}/ARkitScenes/metadata/train/arkitscenes_metadata_train.json",
                "r",
            ) as f:
                metadata = json.load(f)

            valid_scenes = sorted([s for s in scenes if s in metadata])
            generated_path = "./generated_paths/arkitscene"

            QA_paris, unavailables = process_dataset(
                valid_scenes,
                generator=lambda scene_name, **_: generate_rt_plan(
                    scene_name,
                    generated_path=generated_path,
                    data=metadata,
                    skip_repeated=True,
                    skip_repeated_items=skip_repeated_items,
                    return_other_info=return_other_info,
                    DEGREE_THRESHOLD1=DEGREE_THRESHOLD1,
                    DEGREE_THRESHOLD2=DEGREE_THRESHOLD2,
                )
            )

        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        print(f"[Summary] {dataset_type} → {len(QA_paris)} QA-pairs, {len(unavailables)} unavailable scenes")
        return QA_paris, unavailables
    


    QA_paris, unavailables = main(
        dataset_type=dataset_type,
        skip_repeated_items=skip_repeated_items,
        return_other_info=return_other_info,
        DEGREE_THRESHOLD1=DEGREE_THRESHOLD1,
        DEGREE_THRESHOLD2=DEGREE_THRESHOLD2
    )

    print('We have total of {} QA-pairs'.format(len(QA_paris)))
    print('We have total of {} unavailables'.format(len(unavailables)))

    with open(f'qa_pairs_{dataset_type}.json', 'w', encoding='utf-8') as f:
        json.dump(QA_paris, f, ensure_ascii=False, indent=4)

