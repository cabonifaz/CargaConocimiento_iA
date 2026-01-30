import os
import urllib.parse
from dataclasses import dataclass

@dataclass 
class DestructuredKey:
    key: str
    extension: str
    base_name: str
    key_prefix: str
    destination_prefix: str

def destruct_key(raw_key: str, parent_prefix: str, destination_parent_prefix: str):
    clean_key = urllib.parse.unquote_plus(raw_key)
    
    return DestructuredKey(
        key = clean_key,
        extension = os.path.splitext(clean_key)[1].lower(),
        base_name = os.path.splitext(os.path.basename(clean_key))[0],
        key_prefix = os.path.dirname(clean_key),
        destination_prefix = os.path.dirname(clean_key).replace(parent_prefix, destination_parent_prefix, 1)
    )