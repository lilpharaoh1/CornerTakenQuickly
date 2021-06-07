import numpy as np
import math

def calc_end_pos(start_pos, distance, num):
    theta_sign = -1.0
    if num < 540:
        theta_sign = 1.0
        theta = math.radians(((num / 540) * 90))
        delta_x = (distance * math.cos(theta)) * theta_sign
    else: 
        num -= 540
        theta = math.radians(90 - ((num / 540) * 90))
        delta_x = (distance * math.cos(theta)) * theta_sign

    
    delta_y = (distance * math.sin(theta)) * -1

    delta_y = int(delta_y * 10)
    delta_x = int(delta_x * 10)

    end_pos = (start_pos[0] + delta_x, start_pos[1] + delta_y)
    return end_pos
