from typing import List, Tuple, Union, Dict


class Slope(object):
    __slots__ = ('rise', 'run')
    def __init__(self, rise: int, run: int):
        self.rise = rise
        self.run = run

class Point(object):
    __slots__ = ('x', 'y')
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def copy(self) -> 'Point':
        return Point(self.x, self.y)

    def to_dict(self) -> Dict[str, int]:
        return {'x':self.x, 'y':self.y}

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> 'Point':
        return cls(d['x'], d['y'])

    def __eq__(self, other: Union['Point', Tuple[int, int]]) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return other[0] == self.x and other[1] == self.y
        elif isinstance(other, Point) and self.x == other.x and self.y == other.y:
            return True
        return False

    def __sub__(self, other: Union['Point', Tuple[int, int]]) -> 'Point':
        if isinstance(other, tuple) and len(other) == 2:
            return Point(self.x - other[0], self.y - other[1])
        elif isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        return None

    def __rsub__(self, other: Tuple[int, int]):
        return Point(other[0] - self.x, other[1] - self.y)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __str__(self) -> str:
        return '({}, {})'.format(self.x, self.y)