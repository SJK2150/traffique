import json

# Read the JSON
with open('mydata/label_studio_tasks.json', 'r') as f:
    tasks = json.load(f)

# Update paths to just filenames (for direct upload)
for task in tasks:
    old_path = task['data']['image']
    # Extract just the filename
    filename = old_path.split('/')[-1]
    # Use simple relative path
    task['data']['image'] = f'images/{filename}'

# Save updated JSON
with open('mydata/label_studio_tasks_simple.json', 'w') as f:
    json.dump(tasks, f, indent=2)

print("✅ Created: mydata/label_studio_tasks_simple.json")
print("   Now upload BOTH:")
print("   1. The images folder (mydata/images/)")
print("   2. The JSON file (mydata/label_studio_tasks_simple.json)")
