import re

def parse_structure_from_text(text):
    lines = text.split('\n')
    structure = {}
    current_path = []

    for line in lines:
        depth = len(re.match(r'\s*', line).group())
        name = line.strip().rstrip('/')

        while len(current_path) > depth:
            current_path.pop()

        if name:
            current_dict = structure
            for path in current_path:
                current_dict = current_dict[path]
            if line.endswith('/'):
                current_dict[name] = {}
                current_path.append(name)
            else:
                current_dict[name] = None

    return structure