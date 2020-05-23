import json

with open('model.json', 'r') as f:
    a = json.load(f)

with open('model_copy.json', 'w') as f:
    json.dump(a, f, indent=4)
