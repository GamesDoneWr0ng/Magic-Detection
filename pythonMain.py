import numpy as np
import pygame as pg
from typing import Self
from enum import IntEnum
from typing import Callable, List
from scipy.optimize import linear_sum_assignment

pg.init()

# Line detection parameters
BRUSH_WIDTH = 5 # rendering
MIN_DISTANCE = 5 # minimum distance between sample points
CIRCLE_ACCEPTANCE = 10 # allowed distance to perfect circle
LINE_ACCEPTANCE = 5 # allowed distance to perfect line
MAX_RADII = 300 # max circle radius to stop straight lines from becoming circles
ANGLE_ACCEPTANCE = np.pi * 0.25 # create new line if angle to last move is too great (corners)
CONNECTION_DISTANCE = 20

# Line merge parameters
MERGE_DISTANCE = 10 # max distance between straight lines
MERGE_ANGLE = 0.1 # max angle between straight lines
CENTER_DISTANCE = 20 ** 2 # max distance between circle centers
MERGE_RADIUS = 20 # max difference in radii
CIRCLE_MERGE_DISTANCE = np.pi * 0.1 # max angle between circle segments

# Rune detection parameters
START_BONUS = 30 # How far away from the circle lines in the start-rune can be from the circle
CENTER_OFFSET = 40**2 # How far away the lines in the start-rune can be from the center
START_ANGLE_ACCEPTANCE = np.pi * 0.4 # Accepted angle between lines in the start-rune
RUNE_CONNECTION_DISTANCE = 40

SIZE = 600 # Screen size
SCREENSIZE = (SIZE, SIZE)

class Line:
    def __init__(self, start: np.ndarray, end: np.ndarray, center: None | np.ndarray, id: int, angle: None | float = None) -> None:
        self._center = center
        self._angle = angle
        self._end = end.astype(np.float64)
        self._start = start.astype(np.float64)
        self._id = id

        self._radius: float = self._getRadius()

        self._touches: dict[int, bool] = dict()

    #region Getters
    @property
    def start(self) -> np.ndarray:
        return self._start

    @start.setter
    def start(self, start: np.ndarray) -> None:
        self._start = start
        self._radius: float = self._getRadius()

    @property
    def end(self) -> np.ndarray:
        return self._end

    @end.setter
    def end(self, end: np.ndarray) -> None:
        self._end = end.astype(np.float64)
        self._radius: float = self._getRadius()

    @property
    def center(self) -> None | np.ndarray:
        return self._center

    @center.setter
    def center(self, center: np.ndarray) -> None:
        self._center = center
        self._radius: float = self._getRadius()

    @property
    def angle(self) -> float:
        return self._angle

    @angle.setter
    def angle(self, angle: float) -> None:
        self._angle = angle

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, id: int) -> None:
        self._id = id
#endregion

    def _getRadius(self) -> float:
        if self._center is not None:
            return (np.linalg.norm(self._start - self._center) + np.linalg.norm(self._end - self._center)) * 0.5
        else:
            return 0

    def connect(self, other: Self) -> bool:
        if self._center is None and other._center is None:
            a = self._end - self._start
            b = other._end - other._start
            a_norm = a / np.linalg.norm(a)
            b_norm = b / np.linalg.norm(b)

            if np.abs(np.dot(a_norm, b_norm)) < MERGE_ANGLE:
                return False

            norm = (a + b) / np.linalg.norm(a + b)

            v = max(other._start - self._start, other._end - self._start, key=lambda i: distanceSquared(self._start, i))
            # v = other._start - self._start
            perp_dist = np.linalg.norm(v - np.dot(v, norm) * norm)
            if perp_dist > MERGE_DISTANCE:
                return False

            def proj(point: np.ndarray) -> float:
                return np.dot(point - self._start, norm)

            a_proj = sorted([proj(self._start), proj(self._end)])
            b_proj = sorted([proj(other._start), proj(other._end)])

            if a_proj[1] + MERGE_DISTANCE < b_proj[0] or b_proj[1] + MERGE_DISTANCE < a_proj[0]:
                return False

            combined_min = min(a_proj[0], b_proj[0])
            combined_max = max(a_proj[1], b_proj[1])

            print(self, "\n", other)

            self._start = self._start + combined_min * norm
            self._end   = self._start + combined_max * norm
            return True

        elif self._center is not None and other._center is not None:
            if distanceSquared(self._center, other._center) > CENTER_DISTANCE:
                return False

            center = (self._center + other._center) * 0.5

            r_a = np.linalg.norm(self._start - self._center)
            r_b = np.linalg.norm(other._start - other._center)

            if np.abs(r_a - r_b) > MERGE_RADIUS:
                return False

            r = (r_a + r_b) * 0.5

            def point_angle(p):
                v = p - center
                return np.arctan2(v[1], v[0]) % (2 * np.pi)

            a1 = point_angle(self._start)
            b1 = point_angle(other._start)

            def arc_range(start, ang):
                """Return sorted range of angles covered by arc, accounting for direction."""
                end = (start + ang) % (2 * np.pi)
                if ang >= 0:
                    return start, end
                else:
                    return end, start

            a_min, a_max = arc_range(a1, self._angle)
            b_min, b_max = arc_range(b1, other._angle)

            def interval_contains(a_min, a_max, b_min, b_max):
                def to_interval(mn, mx):
                    if mx < mn:
                        return [(mn, 2 * np.pi), (0, mx)]
                    return [(mn, mx)]

                a_int = to_interval(a_min, a_max)
                b_int = to_interval(b_min, b_max)
                for aa in a_int:
                    for bb in b_int:
                        if aa[1] + CIRCLE_MERGE_DISTANCE >= bb[0] and bb[1] + CIRCLE_MERGE_DISTANCE >= aa[0]:
                            return True
                return False

            if not interval_contains(a_min, a_max, b_min, b_max):
                return False

            if a_min > a_max:
                a_max += 2*np.pi
            if b_min > b_max:
                b_max += 2*np.pi

            combined_min = min(a_min, b_min)
            combined_max = max(a_max, b_max)
            combined_angle = np.clip(combined_max - combined_min, -2*np.pi, 2*np.pi)

            self._start = center + r * np.array([np.cos(combined_min), np.sin(combined_min)])
            self._end = center + r * np.array([np.cos(combined_max), np.sin(combined_max)])
            self._center = center
            self._angle = combined_angle
            return True

        return False

    def distanceTo(self, point: np.ndarray) -> float:
        if self._center is None:
            lengthSquared = distanceSquared(self._start, self._end)
            t = np.clip(np.dot(point-self._start, (self._end - self._start)) / lengthSquared, 0, 1)
            proj = self._start + t * (self._end - self._start)

            return np.linalg.norm(point - proj)

        else:
            v_start = self._start - self._center
            v_p = point - self._center
            d = np.linalg.norm(v_p)

            if d == 0:
                return self._radius

            v_start_n = v_start / self._radius
            v_p_n = v_p / d

            ang_to_p = signed_angle(v_start_n, v_p_n)


            if (self._angle >= 0 and 0 <= ang_to_p <= self._angle) or (self._angle < 0 and self._angle <= ang_to_p <= 0):
                return abs(d - self._radius)
            else:
                end_angle = np.array([
                    np.cos(self._angle) * v_start_n[0] - np.sin(self._angle) * v_start_n[1],
                    np.sin(self._angle) * v_start_n[0] + np.cos(self._angle) * v_start_n[1]
                ])
                end = self._center + self._radius * end_angle
                return min(np.linalg.norm(point - self._start), np.linalg.norm(point - end))

    def isTouching(self, other: Self) -> bool:
        if other._id in self._touches:
            return self._touches[other._id]

        def within_arc(c, start, angle, point):
            """Check if a point lies on the arc defined by center/start/angle"""
            v_start = start - c
            v_point = point - c
            if not np.isclose(np.linalg.norm(v_point), np.linalg.norm(v_start)):
                return False

            v_start /= np.linalg.norm(v_start)
            v_point /= np.linalg.norm(v_point)

            theta = signed_angle(v_start, v_point)

            if angle >= 0:
                return 0 <= theta <= angle
            else:
                return angle <= theta <= 0

        if self._center is None and other._center is None:
            def orientation(a, b, c):
                """Returns orientation of triplet (a, b, c):
                0 -> collinear, 1 -> clockwise, 2 -> counterclockwise"""
                val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
                if np.isclose(val, 0):
                    return 0
                return 1 if val > 0 else 2

            def on_segment(a, b, c):
                """Check if point b lies on segment a->c"""
                return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
                        min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))

            o1 = orientation(self._start, self._end, other._start)
            o2 = orientation(self._start, self._end, other._end)
            o3 = orientation(other._start, other._end, self._start)
            o4 = orientation(other._start, other._end, self._end)

            if o1 != o2 and o3 != o4:
                self._touches[other._id] = True
                other._touches[self._id] = True
                return True

            if ((o1 == 0 and on_segment(self._start,  other._start, self._end))
             or (o2 == 0 and on_segment(self._start,  other._end,   self._end))
             or (o3 == 0 and on_segment(other._start, self._start,  other._end))
             or (o4 == 0 and on_segment(other._start, self._end,    other._end))):
                self._touches[other._id] = True
                other._touches[self._id] = True
                return True

            self._touches[other._id] = False
            other._touches[self._id] = False
            return False

        elif self._center is not None and other._center is not None:
            d = np.linalg.norm(other._center - self._center)

            if d > self._radius + other._radius or d < abs(self._radius - other._radius) or np.isclose(d, 0):
                self._touches[other._id] = False
                other._touches[self._id] = False
                return False

            a = (self._radius ** 2 - other._radius ** 2 + d ** 2) / (2 * d)
            h_sq = self._radius ** 2 - a ** 2
            if h_sq < 0:
                self._touches[other._id] = False
                other._touches[self._id] = False
                return False

            h = np.sqrt(h_sq)
            p = self._center + a * (other._center - self._center) / d  # Base point along the center line
            offset = h * np.array([-(other._center - self._center)[1], (other._center - self._center)[0]]) / d

            intersections = [p + offset, p - offset]

            for pt in intersections:
                if within_arc(self._center, self._start, self._angle, pt) and within_arc(other._center, other._start, other._angle, pt):
                    self._touches[other._id] = True
                    other._touches[self._id] = True
                    return True

            self._touches[other._id] = False
            other._touches[self._id] = False
            return False

        else:
            if self._center is None:
                p1 = self._start
                p2 = self._end
                c  = other._center
                p  = other._start
                a  = other._angle
                r = other._radius
            else:
                p1 = other._start
                p2 = other._end
                c  = self._center
                p  = self._start
                a  = self._angle
                r =  self._radius

            d = p2 - p1
            f = p1 - c

            A = np.dot(d, d)
            B = 2 * np.dot(f, d)
            C = np.dot(f, f) - r ** 2

            disc = B ** 2 - 4 * A * C
            if disc < 0:
                self._touches[other._id] = False
                other._touches[self._id] = False
                return False # No intersection

            sqrt_disc = np.sqrt(disc)
            t1 = (-B - sqrt_disc) / (2 * A)
            t2 = (-B + sqrt_disc) / (2 * A)
            ts = [t for t in [t1, t2] if 0 <= t <= 1] # within line segment

            if not ts:
                self._touches[other._id] = False
                other._touches[self._id] = False
                return False

            for t in ts:
                pt = p1 + t * d
                if within_arc(c, p, a, pt):
                    self._touches[other._id] = True
                    other._touches[self._id] = True
                    return True

            self._touches[other._id] = False
            other._touches[self._id] = False
            return False

    def getBounds(self) -> np.ndarray:
        if self._center is None:
            return np.array([[min(self._start[0], self._end[0]), min(self._start[1], self._end[1])],
                             [max(self._start[0], self._end[0]), max(self._start[1], self._end[1])]])
        else:
            return np.array([[self._center[0] - self._radius, self._center[1] - self._radius],
                             [self._center[0] + self._radius, self._center[1] + self._radius],])

    def norm(self, bounds: np.ndarray) -> None:
        size = bounds[1] - bounds[0]
        self._start = (self._start - bounds[0]) / size
        self._end = (self._end - bounds[0]) / size
        if self._center is not None:
            self._center = (self._center - bounds[0]) / size
            self._radius /= size
            if self._angle < 0:
                self._end = self._start
                rotation = np.array([[np.cos(self._angle), -np.sin(self._angle)],
                                         [np.sin(self._angle), np.cos(self._angle)]], dtype=np.float64)
                self._start = np.dot((self._start - self._center), rotation) + self._center
                self._angle *= -1

    def cost(self, other: Self) -> float:
        if self._center is None and other._center is None:
            return np.sqrt(min(
                    distanceSquared(self._start, other._start) + distanceSquared(self._end, other._end),
                    distanceSquared(self._start, other._end) + distanceSquared(self._end, other._start)
            ))
        elif self._center is not None and other._center is not None:
            return (np.linalg.norm(self._center - other._center)
                    + np.linalg.norm(self._start - other._start)
                    + np.abs(self._angle - other._angle) * 0.25)
        else:
            if self._center is None:
                l1 = self._start
                l2 = self._end
                c1 = other._start
                c2 = other._end
                r = other._radius
            else:
                l1 = other._start
                l2 = other._end
                c1 = self._start
                c2 = self._end
                r = self._radius
            penalty = 1 + 2 * np.linalg.norm(l1 - l2) / r
            return (np.sqrt(min(distanceSquared(l1, c1) + distanceSquared(l2, c2),
                               distanceSquared(l1, c2) + distanceSquared(l2, c1)))
                    * penalty)

    def __str__(self) -> str:
        return f"Start: {self._start}, End: {self._end}, Center: {self._center}, Angle: {np.rad2deg(self._angle) if self._angle is not None else None}, Id: {self._id}"



class Connection:
    def __init__(self, start: np.ndarray, end: np.ndarray, points: list[np.ndarray], id: int) -> None:
        self._start = start
        self._end = end
        self._points: np.ndarray = np.array(points)
        self._connections: set[int] = set()
        self._id = id

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, id: int) -> None:
        self._id = id

    def isTouching(self, point: np.ndarray) -> bool:
        return np.any(np.linalg.norm(self._points - point, axis=1) < CONNECTION_DISTANCE)

    def connect(self, other: Self) -> None:
        self ._connections.add(other._id)
        other._connections.add(self._id)

    def __str__(self) -> str:
        return f"Start: {self._start}, End: {self._end}, Points: {len(self._points)}, Connections: {self._connections}, Id: {self._id}"

#region Line Detection
def signed_angle(u, v):
    dot = np.clip(np.dot(u, v), -1.0, 1.0)
    cross = np.cross(u, v)
    return np.arctan2(cross, dot)

def unsigned_angle(a: np.ndarray, b: np.ndarray) -> float:
    # Ensure they are numpy arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Normalize
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        raise ValueError("Input vectors must be non-zero.")

    # Dot product for cosine
    dot = np.dot(a, b) / (a_norm * b_norm)

    # Clamp dot to [-1, 1] to avoid floating point issues
    dot = np.clip(dot, -1.0, 1.0)

    # Angle magnitude
    angle = np.arccos(dot)

    return angle

def best_fit_circle(points) -> None | tuple[bool, np.ndarray, float]:
    """
    Given a list of 2D numpy arrays (points), return the center (x, y)
    of the circle that best fits all points in a least-squares sense.
    Return None if no unique circle exists.
    """
    points = np.array(points)
    if len(points) < 3:
        return None

    x = points[:, 0]
    y = points[:, 1]

    # Set up linear system: A * [D, E, F]^T = b
    A = np.column_stack((x, y, np.ones_like(x)))
    b = -(x**2 + y**2)

    try: # hopefully possible to port to c++
        sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    # If the rank is less than 3, the system is degenerate (collinear points)
    if rank < 3:
        return None

    D, E, F = sol
    cx = -D / 2
    cy = -E / 2

    center = np.array([cx, cy])
    rel = points - center
    angles = np.arctan2(rel[:, 1], rel[:, 0])

    # Unwrap angles to handle continuous rotation across -π/π boundary
    unwrapped = np.unwrap(angles)

    # Total signed angle traversed
    total_angle: float = unwrapped[-1] - unwrapped[0]

    # Clamp to [-2π, 2π] for up to one full rotation
    total_angle = np.clip(total_angle, -2 * np.pi, 2 * np.pi)

    distances = np.sum(rel**2, axis=1)**0.5
    ok = distances[0] < MAX_RADII and (distances.max() - distances.min() < CIRCLE_ACCEPTANCE*2)

    return ok, center, total_angle

def points_near_line(points, max_dist):
    """
    Given a list of 2D numpy arrays (points), determine whether there exists
    a line such that all points are within 'max_dist' from it.

    Returns True if such a line exists, False otherwise.
    """
    points = np.array(points)
    if len(points) < 2:
        return True  # Any line can pass through 0 or 1 points

    # Center the points to remove translation effects
    mean = np.mean(points, axis=0)
    centered = points - mean

    # Singular Value Decomposition (SVD)
    # The first right-singular vector gives the best-fit line direction
    _, _, vh = np.linalg.svd(centered)
    direction = vh[0]

    # Compute perpendicular distances from points to the best-fit line
    # Distance from point p to line through origin in direction v is |p·v_perp|
    v_perp = np.array([-direction[1], direction[0]])
    distances = np.abs(centered @ v_perp)

    all_within = np.all(distances <= max_dist)
    return all_within, mean, direction

def point_within_line_distance(point, line_point, direction, max_dist):
    """
    Check if `point` is within `max_dist` of the infinite line defined by
    `line_point` and `direction`.

    All inputs are 2D numpy arrays.
    """
    direction = direction / np.linalg.norm(direction)  # Ensure it's a unit vector
    v_perp = np.array([-direction[1], direction[0]])  # Perpendicular vector

    # Vector from the line's point to the new point
    delta = point - line_point

    # Perpendicular distance from point to line
    dist = abs(np.dot(delta, v_perp))

    return dist <= max_dist

def getLines(screen: pg.Surface) -> list[Line|Connection]:
    lines: list[Line | Connection] = []
    currentLine: None | Line = None

    drawingConnection: bool = False
    connection: None | int = None

    lastDiff = np.zeros(2)
    points: list[np.ndarray] = []

    isCircle = False
    center: None | np.ndarray = None
    line: None | tuple[np.ndarray, np.ndarray] = None

    angle = 0
    lastPos: np.ndarray = np.zeros(2, dtype=float)
    pos: np.ndarray = np.zeros(2, dtype=float)

    mode = 0

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return lines
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_1:
                    mode = 0
                    drawingConnection = False
                elif event.key == pg.K_2:
                    mode = 1
                    currentLine = None

        keys = pg.key.get_pressed()
        if keys[pg.K_ESCAPE] or keys[pg.K_RETURN]:
            return lines

        drawing = pg.mouse.get_pressed()[0]

        if np.sum((pos-lastPos)**2) > MIN_DISTANCE:
            if drawing:
                pg.draw.line(screen, (0,0,255) if mode == 0 else (0,255,0), lastPos, pos, BRUSH_WIDTH)

                points.append(pos)

                if mode == 1:
                    if not drawingConnection:
                        points = [lastPos, pos]
                        for idx, l in enumerate(lines):
                            if isinstance(l, Connection):
                                if l.isTouching(pos):
                                    connection = idx
                                    break
                        else:
                            connection = None
                    lastPos = pos

                    drawingConnection = True
                    continue

                if currentLine is None:
                    isCircle = False

                    points = [lastPos, pos]

                    currentLine = Line(lastPos, pos, None, len(lines))
                else:
                    diff = pos - lastPos
                    if unsigned_angle(lastDiff, diff) > ANGLE_ACCEPTANCE:
                        if isCircle:
                            currentLine.center = center
                            currentLine.angle = angle
                        currentLine.end = pos
                        lines.append(currentLine)
                        currentLine = None
                        isCircle = False
                        center = None
                        line = None
                        angle = 0
                        continue

                    if not isCircle:
                        if line is None:
                            ok, *line = points_near_line(points, LINE_ACCEPTANCE)
                            isCircle = not ok
                        else:
                            if not point_within_line_distance(pos, line[0], line[1], LINE_ACCEPTANCE):
                                ok, *line = points_near_line(points, LINE_ACCEPTANCE)
                                isCircle = not ok

                    if isCircle:
                        if center is None:
                            result = best_fit_circle(points)
                            if result is not None:
                                ok, center, angle = result
                                drawing = ok # newline if not ok
                                isCircle = ok # isLine if not ok
                            else:
                                drawing = False
                                isCircle = False
                        else:
                            if np.sum((pos-center)**2) > CIRCLE_ACCEPTANCE**2:
                                result = best_fit_circle(points)
                                if result is not None:
                                    drawing, center, angle = result
                                else:
                                    drawing = False
            lastDiff = pos - lastPos
            lastPos = pos

        if not drawing and drawingConnection:
            lines.append(Connection(points[0], points[-1], points, len(lines)))
            for l in lines[:-1]:
                if isinstance(l, Connection):
                    if l.isTouching(pos):
                        lines[-1].connect(l)
                        break
            if connection is not None:
                lines[-1].connect(lines[connection])
            connection = None
            drawingConnection = False

        if not drawing and currentLine is not None:
            if isCircle:
                currentLine.center = center
                currentLine.angle = angle

            currentLine.end = pos
            lines.append(currentLine)
            currentLine = None
            isCircle = False
            center = None
            line = None
            angle = 0

        pos = np.array(pg.mouse.get_pos())
        pg.display.flip()
    return lines

#endregion

#region Rune Detection
"""
IDEAS:
Have start rune spesify amount of registers/memory prob. registers
Circle divided like a pizza, one slice per register
"""
class RuneType(IntEnum):
    START = 0
    PATH = 1
    BRANCH = 2
    OPERATOR = 3
    MEMORY = 4

    FIRE = 10

class Rune:
    def __init__(self, runeType: RuneType, tier: int, id: int, lines: list[int], nextRune: None | Self = None) -> None:
        self._type = runeType
        self._tier = tier
        self._id = id
        self._lines = lines
        self._next = nextRune

    @property
    def lines(self) -> list[int]:
        return self._lines

    def __str__(self) -> str:
        return f"{self._type.name} {self._tier} {self._id} {self._lines} {self._next}"

    def __eq__(self, other: Self) -> bool:
        return self._type == other._type & self._tier == other._tier

Pattern = List[Line]
class PatternMatcher:
    def __init__(self):
        self._patterns: dict[str, Pattern] = dict()
        self._runes: dict[str, Rune] = dict()
        self._similarity_func: Callable[[Pattern, Pattern], float] = self._default_similarity

    def add_pattern(self, name: str, pattern: Pattern, rune: Rune) -> None:
        self._patterns[name] = pattern
        self._runes[name] = rune

    def match(self, input_vectors: Pattern) -> tuple[Rune, float]:
        best_name = None
        best_score = float('-inf')
        for name, pattern in self._patterns.items():
            if len(input_vectors) < len(pattern):
                continue
            score = self._similarity_func(input_vectors, pattern)
            if score > best_score:
                best_name = name
                best_score = score
        return self._runes[best_name], best_score

    def _default_similarity(self, a: Pattern, b: Pattern) -> float:
        """Hungarian algorithm, c++ implementation on wikipedia pog"""
        n, m = len(a), len(b)

        # Compute cost matrix: pairwise distances between all vectors
        cost = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                cost[i, j] = a[i].cost(b[j])

        # Solve the optimal assignment (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost)

        # Average cost of matched pairs
        mean_dist = cost[row_ind, col_ind].mean()

        # Penalty for extra or missing lines
        unmatched_penalty = abs(n - m) * cost.mean()

        total_cost = mean_dist + unmatched_penalty
        return -total_cost

def distanceSquared(a: np.ndarray, b: np.ndarray) -> float:
    return ((a-b)**2).sum()

def mergeLines(lines: list[Line|Connection]) -> list[Line|Connection]:
    i = 0
    while i < len(lines):
        if isinstance(lines[i], Connection):
            i += 1
            continue
        for j in range(i+1, len(lines)):
            if isinstance(lines[j], Connection):
                continue
            if lines[i].connect(lines[j]):
                lines.pop(j)
                break
        else:
            i += 1

    for idx, line in enumerate(lines):
        line.id = idx
    return lines

def testMerge(lines: list[Line]) -> list[int]:
    result = []
    i = 0
    while i < len(lines):
        for j in range(i+1, len(lines)):
            if lines[i].connect(lines[j]):
                result.append(i)
                break
        i += 1
    return result

def linesWithinCircle(lines: list[Line], center: np.ndarray, radius: float) -> list[int]:
    result: list[int] = []
    for idx, line in enumerate(lines):
        if line.center is not None:
            continue
        if distanceSquared(center, line.start) <= radius and distanceSquared(center, line.end) <= radius:
            result.append(idx)
    return result

def linesTouchingCircle(lines: list[Line], center: np.ndarray, radius: float, tolerance: float) -> list[int]:
    result: list[int] = []
    for idx, line in enumerate(lines):
        if isinstance(line, Line) and line.center is not None:
            continue
        if np.abs(np.linalg.norm(center - line.start) - radius) <= tolerance and np.abs(np.linalg.norm(center - line.end) - radius) <= tolerance:
            result.append(idx)
    return result

def detectRune(lines: list[Line], used: set[int], start: np.ndarray) -> None | Rune:
    startLine: Line = min(filter(lambda l: isinstance(l, Line), lines), key=lambda l: l.distanceTo(start))
    if startLine.distanceTo(start) > RUNE_CONNECTION_DISTANCE:
        return None
    current = [startLine]
    used.add(startLine.id)
    found = True
    while found:
        found = False
        for line in lines:
            if line.id in used:
                continue

            for k in current:
                if k == line:
                    continue
                if k.isTouching(line):
                    current.append(line)
                    used.add(line.id)
                    found = True
                    break

    if len(current) == 0:
        return None

    bounds = np.array([[0,0], [np.inf, np.inf]])
    for i in current:
        b = i.getBounds()
        bounds[0][bounds[0] > b[0]] = b[0][bounds[0] > b[0]]
        bounds[1][bounds[1] < b[1]] = b[1][bounds[1] < b[1]]

    bounds = np.array([np.min(bounds[0]), np.max(bounds[1])])
    for i in current:
        i.norm(bounds)

    return None

def detectRunes(lines: list[Line|Connection]) -> list[Rune]:
    runes: list[Rune] = []

    lines = mergeLines(lines)
    used: set[int] = set()

    # find start
    for idx, line in enumerate(lines):
        if isinstance(line, Connection):
            continue
        if line.center is not None and np.abs(line.angle) >= 2 * np.pi - 1e-3:
            within = linesTouchingCircle(lines, line.center, line.radius, START_BONUS)
            if len(within) >= 2:
                angles = []
                startLines = [idx]
                for i in within:
                    if lines[i].distanceTo(line.center) < CENTER_OFFSET:
                        v = lines[i].end - lines[i].start
                        if v[1] < 0:
                            v *= -1

                        angles.append(np.arctan2(v[1], v[0]))
                        #angles.append(getAngle(lines[i].start, lines[i].end))
                        startLines.append(i)
                if len(angles) < 2:
                    continue
                angles.sort()
                diffs = [(angles[(i+1) % len(angles)] - angles[i]) % np.pi for i in range(len(angles))]
                if max(diffs) - min(diffs) < START_ANGLE_ACCEPTANCE / len(angles):
                    runes.append(Rune(RuneType.START, len(angles) - 2, len(runes), startLines))
                    break
    if len(runes) == 0:
        return runes

    print(runes[0])

    used.update(runes[0].lines)

    return runes

#endregion

def main() -> None:
    screen = pg.display.set_mode(SCREENSIZE)
    lines = getLines(screen)
    detectRunes(lines)

    merged = [] #testMerge(lines)
    # print(merged)

    # debug rendering
    for i in lines:
        print(i)

    for idx, line in enumerate(lines):
        if isinstance(line, Connection):
            continue
        if line.center is not None:
            pg.draw.line(screen, (255,0,0), line.start, line.end, 3)

            center = line.center
            pg.draw.circle(screen, (255,0,0), center, 3, 3)
            pos = line.start - center
            angleGoal = line.angle
            angle = np.deg2rad(2) * np.sign(angleGoal)
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                         [np.sin(angle), np.cos(angle)]], dtype=np.float64)

            totalAngle = 0
            while totalAngle < angleGoal if angleGoal > 0 else totalAngle > angleGoal:
                lastPos = pos
                pos = np.dot(rotation, pos)
                pg.draw.line(screen, (255,255,255) if idx not in merged else (0,255,0), lastPos+center, pos+center, 1)
                totalAngle += angle
        else:
            pg.draw.line(screen, (255,255,255), line.start, line.end, 1)

    pg.display.flip()
    while True:
        events = pg.event.get()
        for event in events:
            if event.type == pg.QUIT:
                return

if __name__ == "__main__":
    main()