import re
from typing import Dict, Any

def parse_structure_from_text(text: str) -> Dict[str, Any]:
    lines = text.split('\n')
    structure = {}
    path_stack = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        indent = len(line) - len(line.lstrip())
        name = stripped.rstrip('/')

        while len(path_stack) > indent // 2:
            path_stack.pop()

        if stripped.endswith('/'):
            current = structure
            for path in path_stack:
                current = current[path]
            current[name] = {}
            path_stack.append(name)
        else:
            current = structure
            for path in path_stack:
                current = current[path]
            current[name] = None

    return structure

def validate_structure(structure: Dict[str, Any]) -> bool:
    def is_valid_name(name: str) -> bool:
        return bool(re.match(r'^[a-zA-Z0-9_.-]+$', name))

    def validate_recursive(node: Dict[str, Any]) -> bool:
        for name, value in node.items():
            if not is_valid_name(name):
                return False
            if isinstance(value, dict):
                if not validate_recursive(value):
                    return False
            elif value is not None:
                return False
        return True

    return validate_recursive(structure)

def structure_to_text(structure: Dict[str, Any], indent: int = 0) -> str:
    result = []
    for name, value in structure.items():
        if isinstance(value, dict):
            result.append("  " * indent + name + "/")
            result.append(structure_to_text(value, indent + 1))
        else:
            result.append("  " * indent + name)
    return "\n".join(result)
