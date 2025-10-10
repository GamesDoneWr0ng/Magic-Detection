import numpy as np
import pygame as pg

pg.init()

BRUSH_WIDTH = 5 # rendering
MIN_DISTANCE = 5 # minimum distance between sample points
CIRCLE_ACCEPTANCE = 10 # allowed distance to perfect circle
LINE_ACCEPTANCE = 5 # allowed distance to perfect line
MAX_RADII = 300 # max circle radius to stop straight lines from becoming circles
ANGLE_ACCEPTANCE = np.pi * 0.25 # create new line if angle to last move is too great (corners)

SIZE = 600
SCREENSIZE = (SIZE, SIZE)

class Line:
    def __init__(self, start: np.ndarray, end: np.ndarray, center: None | np.ndarray, angle: None | float = None) -> None:
        self._center = center
        self._angle = angle
        self._end = end
        self._start = start
        self._length = 0

        #self._startVector = self._end - self._start

    @property
    def start(self) -> np.ndarray:
        return self._start

    @property
    def end(self) -> np.ndarray:
        return self._end

    @end.setter
    def end(self, end: np.ndarray) -> None:
        self._end = end

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

    # @property
    # def startVector(self) -> np.ndarray:
    #     return self._startVector

    def __str__(self) -> str:
        return f"Start: {self._start}, End: {self._end}, Center: {self._center}, Angle: {np.rad2deg(self._angle) if self._angle is not None else None}"

#%% Line Detection
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


#%% Magic detection
def detectSpell(lines: list[Line]) -> tuple[str, float] | None:
    pass

def main() -> None:
    screen = pg.display.set_mode(SCREENSIZE)
    lines = getLines(screen)
    detectSpell(lines)

    # debug rendering
    for i in lines:
        print(i)

    for line in lines:
        if line.center is not None:
            pg.draw.line(screen, (255,0,0), line.start, line.end, 3)

            center = line.center
            pos = line.start - center
            angleGoal = line.angle
            angle = np.deg2rad(2) * np.sign(angleGoal)
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                         [np.sin(angle), np.cos(angle)]], dtype=np.float64)

            totalAngle = 0
            while totalAngle < angleGoal if angleGoal > 0 else totalAngle > angleGoal:
                lastPos = pos
                pos = np.dot(rotation, pos)
                pg.draw.line(screen, (255,255,255), lastPos+center, pos+center, 1)
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