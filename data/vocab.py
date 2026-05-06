# ----------- Mapping char to index -----------

import string

VIET_CHARS = list(
    "aàáạảãâầấậẩẫăằắặẳẵ"
    "eèéẹẻẽêềếệểễ"
    "iìíịỉĩ"
    "oòóọỏõôồốộổỗơờớợởỡ"
    "uùúụủũưừứựửữ"
    "yỳýỵỷỹ"
    "đ"
)

ALL_CHARS = list(string.digits + string.ascii_lowercase) + VIET_CHARS + [" "]

char2idx = {c: i + 1 for i, c in enumerate(ALL_CHARS)}  # 0 = blank
idx2char = {i: c for c, i in char2idx.items()}