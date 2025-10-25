import numpy as np
import pygame as pg
from typing import Self
from enum import IntEnum

pg.init()

# Line detection parameters
BRUSH_WIDTH = 5 # rendering
MIN_DISTANCE = 5 # minimum distance between sample points
CIRCLE_ACCEPTANCE = 10 # allowed distance to perfect circle
LINE_ACCEPTANCE = 5 # allowed distance to perfect line
MAX_RADII = 300 # max circle radius to stop straight lines from becoming circles
ANGLE_ACCEPTANCE = np.pi * 0.25 # create new line if angle to last move is too great (corners)

# Rune detection parameters
MERGE_DISTANCE = 10
MERGE_ANGLE = 0.35
CENTER_DISTANCE = 40 ** 2
MERGE_RADIUS = 20
CIRCLE_MERGE_DISTANCE = np.pi * 0.1

SIZE = 600
SCREENSIZE = (SIZE, SIZE)

class Line:
    def __init__(self, start: np.ndarray, end: np.ndarray, center: None | np.ndarray, angle: None | float = None) -> None:
        self._center = center
        self._angle = angle
        self._end = end.astype(np.float64)
        self._start = start.astype(np.float64)
        self._length = 0

    #region Getters
    @property
    def start(self) -> np.ndarray:
        return self._start

    @property
    def end(self) -> np.ndarray:
        return self._end

    @end.setter
    def end(self, end: np.ndarray) -> None:
        self._end = end.astype(np.float64)

    @property
    def center(self) -> None | np.ndarray:
        return self._center

    @center.setter
    def center(self, center: np.ndarray) -> None:
        self._center = center

    @property
    def length(self) -> float:
        return self._length

    @length.setter
    def length(self, length: float) -> None:
        self._length = length

    @property
    def angle(self) -> float:
        return self._angle

    @angle.setter
    def angle(self, angle: float) -> None:
        self._angle = angle
#endregion

    def connect(self, other: Self) -> bool:
        if self._center is None and other._center is None:
            a_norm = (self._end - self._start) / np.linalg.norm(self._end - self._start)
            b_norm = (other._end - other._start) / np.linalg.norm(other._end - other._start)

            if np.abs(np.dot(a_norm, b_norm)) < MERGE_ANGLE:
                return False

            v = max(other._start - self._start, other._end - self._start, key=lambda i: distanceSquared(self._start, i))
            #v = other._start - self._start
            perp_dist = np.linalg.norm(v - np.dot(v, a_norm) * a_norm)
            if perp_dist > MERGE_DISTANCE:
                return False

            def proj(point: np.ndarray) -> float:
                return np.dot(point - self._start, a_norm)

            a_proj = sorted([proj(self._start), proj(self._end)])
            b_proj = sorted([proj(other._start), proj(other._end)])

            if a_proj[1] + MERGE_DISTANCE < b_proj[0] or b_proj[1] + MERGE_DISTANCE < a_proj[0]:
                return False

            combined_min = min(a_proj[0], b_proj[0])
            combined_max = max(a_proj[1], b_proj[1])

            self._start += combined_min * a_norm
            self._end = self._start + combined_max * a_norm
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

    def __str__(self) -> str:
        return f"Start: {self._start}, End: {self._end}, Center: {self._center}, Angle: {np.rad2deg(self._angle) if self._angle is not None else None}"

#region Line Detection
def getAngle(a: np.ndarray, b: np.ndarray) -> float:
    """
    Returns the signed angle (in radians) between two 2D vectors.
    Positive = counter-clockwise (left turn)
    Negative = clockwise (right turn)
    """
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

    # Cross product in 2D (determinant form)
    # cross = a[0] * b[1] - a[1] * b[0]

    # Sign from cross product
    # if cross < 0:
    #     angle = -angle

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

def getLines(screen: pg.Surface) -> list[Line]:
    lines: list[Line] = []
    currentLine: None | Line = None

    #angles: list[float] = []
    lastDiff = np.zeros(2)
    points: list[np.ndarray] = []

    isCircle = False
    center: None | np.ndarray = None
    line: None | tuple[np.ndarray, np.ndarray] = None

    angle = 0
    lastPos: np.ndarray = np.zeros(2, dtype=float)
    pos: np.ndarray = np.zeros(2, dtype=float)

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return lines

        keys = pg.key.get_pressed()
        if keys[pg.K_ESCAPE] or keys[pg.K_RETURN]:
            return lines

        drawing = pg.mouse.get_pressed()[0]

        if np.sum((pos-lastPos)**2) > MIN_DISTANCE:
            if drawing:
                pg.draw.line(screen, (0,0,255), lastPos, pos, BRUSH_WIDTH)

                points.append(pos)

                if currentLine is None:
                    isCircle = False

                    points = [lastPos, pos]

                    currentLine = Line(lastPos, pos, None)
                else:
                    diff = pos - lastPos
                    if getAngle(lastDiff, diff) > ANGLE_ACCEPTANCE:
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

    def __eq__(self, other: Self) -> bool:
        return self._type == other._type & self._tier == other._tier

def distanceSquared(a: np.ndarray, b: np.ndarray) -> float:
    return ((a-b)**2).sum()

def mergeLines(lines: list[Line]) -> list[Line]:
    i = 0
    while i < len(lines):
        for j in range(i+1, len(lines)):
            if lines[i].connect(lines[j]):
                lines.pop(j)
                break
        else:
            i += 1
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

def detectRunes(lines: list[Line]) -> list[Rune]:
    runes: list[Rune] = []

    # find start
    lines = mergeLines(lines)


    # points: dict[Node, Node] = dict()
    #
    # for line in lines:
    #     start = Node(line.start)
    #     if start in points:
    #         start = points[start]
    #
    #     if line.center is None:
    #         end = Node(line.end)
    #         if end in points:
    #             end = points[end]
    #
    #         start.addConnection(end, line)
    #
    #     else:


    #rune = Rune(RuneType.START, 0, len(runes))

    return runes

#endregion

#region Magic detection
def detectSpell(lines: list[Line]) -> tuple[str, float] | None:
    pass

#endregion

def main() -> None:
    screen = pg.display.set_mode(SCREENSIZE)
    lines = getLines(screen)
    detectRunes(lines)
    #detectSpell(lines)

    merged = [] #testMerge(lines)
    # print(merged)

    # debug rendering
    for i in lines:
        print(i)

    for idx, line in enumerate(lines):
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