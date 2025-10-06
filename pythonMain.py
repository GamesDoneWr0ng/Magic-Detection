import numpy as np
import pygame as pg

pg.init()

BRUSH_WIDTH = 5
MIN_DISTANCE = 5
ACCEPTANCE = 1

GRIDSIZE = 600
SCREENSIZE = (GRIDSIZE, GRIDSIZE)

class Line:
    def __init__(self, start: np.ndarray, end: np.ndarray, angle: float) -> None:
        self._angle = angle
        self._end = end
        self._start = start
        self._length = 0

        self._startVector = self._end - self._start

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
    def angle(self) -> float:
        return self._angle

    @angle.setter
    def angle(self, angle: float) -> None:
        self._angle = angle

    @property
    def length(self) -> float:
        return self._length

    @length.setter
    def length(self, length: float) -> None:
        self._length = length

    @property
    def startVector(self) -> np.ndarray:
        return self._startVector

    def __str__(self) -> str:
        return f"Start: {self._start}, End: {self._end}, Angle: {self._angle}"

def detectSpell(lines: list[Line]) -> str:
    pass

def getAngle(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    dot = np.dot(a, b) / (a_norm * b_norm)

    angle = np.arccos(dot)

    cross = a[0] * b[1] - a[1] * b[0]
    if cross > 0:
        angle = -angle

    return angle

def adjustLine(angles: list[float]) -> tuple[np.bool_, float]:
    arr = np.array(angles, dtype=float)
    mean = arr.mean()
    return np.any(np.abs(arr - mean) <= ACCEPTANCE), mean

def getLines(screen: pg.Surface) -> list[Line]:
    lines: list[Line] = []
    currentLine: Line = None

    angles: list[float] = []

    length = 0
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

                if currentLine is None:
                    lastDiff = pos - lastPos
                    angles = []
                    length = np.sum(lastDiff**2)**0.5

                    currentLine = Line(lastPos, pos, 0)
                else:
                    diff = pos - lastPos
                    angle = getAngle(lastDiff, diff)
                    lastDiff = diff.copy()

                    angles.append(angle)

                    length += np.sum(diff ** 2) ** 0.5

                    if len(angles) == 0:
                        currentLine.angle = angle

                    if not np.abs(currentLine.angle - angle) < ACCEPTANCE:
                        canAdjust, newAngle = adjustLine(angles)
                        if canAdjust:
                            currentLine.angle = newAngle
                        else:
                            currentLine.length = length
                            currentLine.end = lastPos
                            lines.append(currentLine)
                            currentLine = None
                            angles = []

            lastPos = pos

        if not drawing and currentLine is not None:
            currentLine.length = length
            currentLine.end = pos
            lines.append(currentLine)
            currentLine = None
            angles = []

        pos = np.array(pg.mouse.get_pos())
        pg.display.flip()

def main() -> None:
    screen = pg.display.set_mode(SCREENSIZE)
    lines = getLines(screen)
    for i in lines:
        print(i)

    # screen.fill((0, 0, 0))
    for line in lines:
        pg.draw.line(screen, (255,0,0), line.start, line.end, 3)

        pos = line.start.astype(np.float64).T
        angle = line.angle
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]], dtype=np.float64)
        distance = 0
        vector = line.startVector.astype(np.float64).T
        while distance < line.length:
            pg.draw.line(screen, (255,255,255), pos.T, pos.T+vector.T, 1)
            pos += vector
            vector = np.dot(vector, rotation)

            distance += MIN_DISTANCE

    pg.display.flip()
    while True:
        events = pg.event.get()
        for event in events:
            if event.type == pg.QUIT:
                return

if __name__ == "__main__":
    main()