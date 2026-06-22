import json
from typing import Any, Dict, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    output = []
    f = open(path, 'r')
    line = f.readline()
    while line:
        doc = json.loads(line)
        output.append(doc)
        line = f.readline()
    f.close()
    return output

def write_jsonl(
    docs: List[Dict[str, Any]],
    path: str
) -> None:
    f = open(path, 'w')
    for doc in docs:
        f.write("%s\n" % json.dumps(doc, ensure_ascii=False))
    f.close()

def read_json(_path: str) -> Dict[str, Any]:
    _f = open(_path, "r")
    data = json.loads(_f.read())
    _f.close()
    return data

def write_json(
    data: Dict[str, Any],
    path: str
) -> None:
    f = open(path, "w")
    f.write(json.dumps(data, indent=4))
    f.close()

