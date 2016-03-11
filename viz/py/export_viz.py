import json


# create dictionary of data
data = {}

# create first top level object (root)
data['name'] = "Top Level"
data['parent'] = None
# create an array (list) of children
data['children'] = []

# create second object, child of top level
data['children'].append({})
# set properties for that second level object
data['children'][0]['name'] = "Level 2: A"
data['children'][0]['parent'] = "Top Level"

# create additional children of the second object
data['children'][0]['children'] = []
data['children'][0]['children'].append({})
data['children'][0]['children'][0]['name'] = "Son of A"
data['children'][0]['children'][0]['parent'] = "Level 2: A"
data['children'][0]['children'].append({})
data['children'][0]['children'][1]['name'] = "Daughter of A"
data['children'][0]['children'][1]['parent'] = "Level 2: A"


# create another object that is also child of top level
data['children'].append({})
# set properties for this new object
data['children'][1]['name'] = "Level 2: B"
data['children'][1]['parent'] = "Top Level"



# export the dictionary to a json file
with open('viz/data/decision_tree.json', 'w') as jsonfile:
    json.dump(data, jsonfile)