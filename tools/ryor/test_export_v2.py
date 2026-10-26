"""Keep floor outside the robot map in the room's coverage denominator."""
import unittest
from export_v2 import room_grid
from replay import inside


class CoverageTest(unittest.TestCase):
    def test_room_crossing_map_boundary_is_not_fully_searched(self):
        cells, total = room_grid([[24, 0], [26, 0], [26, 1], [24, 1]],
                                 {'cell_m': 0.1, 'origin': [-25, -25], 'size': 500}, inside)
        self.assertEqual(total, 200)
        self.assertEqual(len(cells), 100)
        self.assertEqual(round(100 * len(cells) / total), 50)


if __name__ == '__main__':
    unittest.main()
