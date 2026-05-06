# ----------- Mapping char to index -----------

import string

# lowercase tiếng Việt
VIET_LOWER = list(
    "aàáạảãâầấậẩẫăằắặẳẵ"
    "eèéẹẻẽêềếệểễ"
    "iìíịỉĩ"
    "oòóọỏõôồốộổỗơờớợởỡ"
    "uùúụủũưừứựửữ"
    "yỳýỵỷỹ"
    "đ"
)

VIET_UPPER = [c.upper() for c in VIET_LOWER]

ALL_CHARS = (
    list(string.digits)
    + list(string.ascii_lowercase)
    + list(string.ascii_uppercase)
    + VIET_LOWER
    + VIET_UPPER
    + [" "]
)

# remove duplicate (rất quan trọng ⚠️)
ALL_CHARS = list(dict.fromkeys(ALL_CHARS))

char2idx = {c: i + 1 for i, c in enumerate(ALL_CHARS)}  # 0 = blank
idx2char = {i: c for c, i in char2idx.items()}