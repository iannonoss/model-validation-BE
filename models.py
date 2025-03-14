from pydantic import BaseModel

class PreprocessRequest(BaseModel):
    split_type: str
    random_state: int
    target: str
    features: list
    test_size: float
