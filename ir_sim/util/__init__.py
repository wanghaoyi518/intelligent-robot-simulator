from ir_sim.util.collision_detection import collision_cir_cir, collision_cir_matrix, collision_cir_seg, collision_seg_matrix, collision_seg_seg

from ir_sim.util.range_detection import range_seg_matrix, range_cir_seg, range_seg_seg

from ir_sim.util.reciprocal_vel_obs import reciprocal_vel_obs

# Mode 7 新增：多边形生成和碰撞检测工具
from ir_sim.util.polygon_generator import PolygonGenerator
from ir_sim.util.collision_detection_polygon import (
    collision_point_polygon, collision_circle_polygon, collision_line_polygon,
    collision_polygon_polygon, collision_robot_polygon, collision_path_polygon,
    check_agent_safe_distance, check_path_safe_distance,
    collision_circle_multiple_polygons, collision_line_multiple_polygons,
    get_safe_spawn_area
)