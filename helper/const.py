from helper.tools import Slope, Point

# Slope (rise, run) = (delta_y, delta_x)
### These lines are defined such that facing "up" would be L0 ###
# Create 16 lines to be able to "see" around
VISION_16 = (
#   L0            L1             L2             L3
    Slope(-1, 0), Slope(-2, 1),  Slope(-1, 1),  Slope(-1, 2),
#   L4            L5             L6             L7      
    Slope(0, 1),  Slope(1, 2),   Slope(1, 1),   Slope(2, 1),
#   L8            L9             L10            L11
    Slope(1, 0),  Slope(2, -1),  Slope(1, -1),  Slope(1, -2),
#   L12           L13            L14            L15
    Slope(0, -1), Slope(-1, -2), Slope(-1, -1), Slope(-2, -1)
)

# Create 8 lines to be able to "see" around
# Really just VISION_16 without odd numbered lines
VISION_8 = tuple([VISION_16[i] for i in range(len(VISION_16)) if i%2==0])

# Create 4 lines to be able to "see" around
# Really just VISION_16 but removing anything not divisible by 4
VISION_4 = tuple([VISION_16[i] for i in range(len(VISION_16)) if i%4==0])


VISION_DICT = {4: VISION_4, 8: VISION_8, 16: VISION_16} # Options are [4, 8, 16]


# Size of apple, and also size of a snake dot
DOT_SIZE = 40

# Number of outputs: 4 - Up, Down, Left, Right
NUM_OUTPUTS = 4