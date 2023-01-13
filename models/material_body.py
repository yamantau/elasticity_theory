from models.material_points import Material_points as mp
from typing import List

class Material_body:
    def __init__(self, material_points: List[mp], time):
        self.material_points = material_points
        self.time = time
