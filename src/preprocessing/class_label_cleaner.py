#!/usr/bin/env python3

import json
from pathlib import Path

LABELS_DIR = "food_dataset/labels"
NOTES_JSON = "food_dataset/notes.json"


def main():
    """This methods iterates through classification categories and
    fixes numerical ranking to start from 0 and up if broken"""

    used_classes = set()
    labels_path = Path(LABELS_DIR)

    for txt_file in labels_path.glob('*.txt'):
        with open(txt_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    used_classes.add(class_id)

    used_classes = sorted(used_classes)
    print(f"Found class IDs in labels: {used_classes}")

    with open(NOTES_JSON, 'r') as f:
        notes = json.load(f)

    class_names = {}
    for cat in notes['categories']:
        class_names[cat['id']] = cat['name']

    print(f"Class names from notes.json: {class_names}")

    id_mapping = {}
    yaml_classes = []

    for new_id, old_id in enumerate(used_classes):
        id_mapping[old_id] = new_id
        class_name = class_names.get(old_id, f"class_{old_id}")
        yaml_classes.append(class_name)
        print(f"  {old_id} -> {new_id}: {class_name}")

    print(f"\nUpdating label files...")
    files_updated = 0

    for txt_file in labels_path.glob('*.txt'):
        new_lines = []
        changed = False

        with open(txt_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    old_class_id = int(parts[0])

                    if old_class_id in id_mapping:
                        new_class_id = id_mapping[old_class_id]
                        parts[0] = str(new_class_id)
                        changed = True

                    new_lines.append(' '.join(parts))

        if changed:
            with open(txt_file, 'w') as f:
                for line in new_lines:
                    f.write(line + '\n')
            files_updated += 1

    print(f"Updated {files_updated} label files")

    print(f"\nUpdating notes.json...")

    new_categories = []
    for new_id, old_id in enumerate(used_classes):
        if old_id in class_names:
            new_categories.append({
                "id": new_id,
                "name": class_names[old_id]
            })

    notes['categories'] = new_categories

    with open(NOTES_JSON, 'w') as f:
        json.dump(notes, f, indent=2)

    print(f"Cleaned notes.json with {len(new_categories)} categories")

    print(f"\nDataset.yaml classes:")
    print("=" * 40)
    print(f"nc: {len(yaml_classes)}")
    print("names:")
    for i, name in enumerate(yaml_classes):
        print(f"  {i}: {name}")
    print("=" * 40)


if __name__ == "__main__":
    main()