import os

labels_dir = "./Dataset/labels"  # adjust this path if needed
total_files_modified = 0
total_labels_modified = 0

for file_name in os.listdir(labels_dir):
    if not file_name.endswith(".txt"):
        continue

    file_path = os.path.join(labels_dir, file_name)
    with open(file_path, "r") as f:
        lines = f.readlines()

    updated_lines = []
    file_modified = False
    labels_modified_in_file = 0

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == '1':
            parts[0] = '0'
            file_modified = True
            labels_modified_in_file += 1
        updated_lines.append(" ".join(parts))

    if file_modified:
        total_files_modified += 1
        total_labels_modified += labels_modified_in_file
        with open(file_path, "w") as f:
            f.write("\n".join(updated_lines))

print(f"\n✅ Summary:")
print(f"→ Total files modified: {total_files_modified}")
print(f"→ Total labels changed from class 1 to 0: {total_labels_modified}")
