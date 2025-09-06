import json
import random
import uuid

def generate_pairs(correct_directions):
    # Define possible directions with "Turn" prefix
    directions = ['Turn Left', 'Turn Right', 'Turn Back']
    
    # Convert correct directions into "Turn ..." format
    correct_answer = ", ".join(correct_directions)
    correct_answer = correct_answer.replace('Right', 'Turn Right')
    correct_answer = correct_answer.replace('Left', 'Turn Left')
    correct_answer = correct_answer.replace('Back', 'Turn Back')
    
    # Decide how many wrong answers to generate
    num_wrong_answers = 2 if len(correct_directions) == 1 else 3
    
    # Generate unique wrong answers
    wrong_answers = set()
    while len(wrong_answers) < num_wrong_answers:
        random_directions = [random.choice(directions) for _ in range(len(correct_directions))]
        wrong_answer = ", ".join(random_directions)
        if wrong_answer != correct_answer:
            wrong_answers.add(wrong_answer)
    
    # Combine correct and wrong answers, then shuffle
    all_answers = list(wrong_answers) + [correct_answer]
    random.shuffle(all_answers)
    
    # Generate option labels (A, B, C, D...)
    options = [f"{chr(65+i)}. {ans}" for i, ans in enumerate(all_answers)]
    
    # Find the correct answer label
    correct_index = all_answers.index(correct_answer)
    answer = chr(65 + correct_index)
    
    return answer, options

def generate_unique_id():
    return str(uuid.uuid4())

if __name__ == '__main__':
    outputs = []
    # List of datasets to merge
    dataset_types = ['arkitscenes', 'scannetpp', 'scannet']
    
    for dataset_type in dataset_types:
        qa_file = f'./qa_pairs_{dataset_type}.json'
        with open(qa_file, 'r') as f:
            data = json.load(f)

        # Build QA pairs for each dataset
        for qa in data:
            scene_name = qa['scene_name']
            description = qa['final_prompt']
            turning_directions = qa['turning_directions']

            answer, options = generate_pairs(turning_directions)

            outputs.append({
                'id': generate_unique_id(),
                'dataset': dataset_type,
                'scene_name': scene_name,
                'question_type': 'route_planning',
                'question': description,
                'answer': answer,
                'options': options
            })

    # Save merged dataset
    with open('./qa_pairs_3datasets.json', 'w') as f:
        json.dump(outputs, f, indent=4)

    print(f"Merged {len(outputs)} QA pairs into ./qa_pairs_3datasets.json")