from pydantic import BaseModel


class Point(BaseModel):
    """Representation of a two-dimensional point coordinate."""

    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        """Computes the distance to another `PointV3`."""
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx**2 + dy**2) ** 0.5


class Triangle(BaseModel):
    """Representation of a triangle"""

    point_a: Point = Point(x=0, y=0)
    point_b: Point = Point(x=1, y=0)
    point_c: Point = Point(x=0, y=1)

    def circumference(self):
        """Circumference of the triangle"""
        return (
            self.point_a.distance_to(self.point_b)
            + self.point_b.distance_to(self.point_c)
            + self.point_c.distance_to(self.point_a)
        )


triangle = Triangle(
    point_a=Point(x=0, y=0),
    point_b=Point(x=0.5, y=0),
    point_c=Point(x=0, y=0.5),
)

print(triangle)