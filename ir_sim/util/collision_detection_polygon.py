import numpy as np
from math import sqrt, cos, sin, pi
from collections import namedtuple
from .polygon_generator import PolygonGenerator

# 定义数据结构
circle = namedtuple('circle', 'x y r')
point = namedtuple('point', 'x y')

def collision_point_polygon(point_pos, polygon_vertices):
    """
    检查点与多边形的碰撞
    Args:
        point_pos: point namedtuple 或 [x, y]
        polygon_vertices: list of [x, y] 多边形顶点
    Returns:
        bool: 是否发生碰撞
    """
    if hasattr(point_pos, 'x'):
        point_coord = [point_pos.x, point_pos.y]
    else:
        point_coord = point_pos
        
    return PolygonGenerator.point_in_polygon(point_coord, polygon_vertices)

def collision_circle_polygon(circle_obj, polygon_vertices):
    """
    检查圆与多边形的碰撞
    Args:
        circle_obj: circle namedtuple (x, y, r)
        polygon_vertices: list of [x, y] 多边形顶点
    Returns:
        bool: 是否发生碰撞
    """
    center = [circle_obj.x, circle_obj.y]
    
    # 如果圆心在多边形内，肯定碰撞
    if PolygonGenerator.point_in_polygon(center, polygon_vertices):
        return True
    
    # 计算圆心到多边形的最小距离
    min_distance = PolygonGenerator.distance_point_to_polygon(center, polygon_vertices)
    
    # 如果最小距离小于半径，则碰撞
    return min_distance < circle_obj.r

def collision_line_polygon(line_start, line_end, polygon_vertices):
    """
    检查线段与多边形的碰撞
    Args:
        line_start: point namedtuple 或 [x, y] 线段起点
        line_end: point namedtuple 或 [x, y] 线段终点
        polygon_vertices: list of [x, y] 多边形顶点
    Returns:
        bool: 是否发生碰撞
    """
    # 转换输入格式
    if hasattr(line_start, 'x'):
        start_coord = [line_start.x, line_start.y]
    else:
        start_coord = line_start
        
    if hasattr(line_end, 'x'):
        end_coord = [line_end.x, line_end.y]
    else:
        end_coord = line_end
    
    return PolygonGenerator.line_intersects_polygon(start_coord, end_coord, polygon_vertices)

def collision_polygon_polygon(poly1_vertices, poly2_vertices):
    """
    检查两个多边形是否碰撞（分离轴定理）
    Args:
        poly1_vertices: list of [x, y] 第一个多边形顶点
        poly2_vertices: list of [x, y] 第二个多边形顶点
    Returns:
        bool: 是否发生碰撞
    """
    # 简化算法：检查顶点是否在对方多边形内
    
    # 检查poly1的顶点是否在poly2内
    for vertex in poly1_vertices:
        if PolygonGenerator.point_in_polygon(vertex, poly2_vertices):
            return True
    
    # 检查poly2的顶点是否在poly1内
    for vertex in poly2_vertices:
        if PolygonGenerator.point_in_polygon(vertex, poly1_vertices):
            return True
    
    # 检查边是否相交
    for i in range(len(poly1_vertices)):
        edge1_start = poly1_vertices[i]
        edge1_end = poly1_vertices[(i + 1) % len(poly1_vertices)]
        
        for j in range(len(poly2_vertices)):
            edge2_start = poly2_vertices[j]
            edge2_end = poly2_vertices[(j + 1) % len(poly2_vertices)]
            
            if PolygonGenerator.line_segments_intersect(edge1_start, edge1_end, 
                                                       edge2_start, edge2_end):
                return True
    
    return False

def collision_robot_polygon(robot_state, robot_radius, polygon_vertices):
    """
    检查机器人（圆形）与多边形的碰撞
    Args:
        robot_state: np.array 机器人状态 [x, y, ...]
        robot_radius: float 机器人半径
        polygon_vertices: list of [x, y] 多边形顶点
    Returns:
        bool: 是否发生碰撞
    """
    robot_circle = circle(robot_state[0, 0], robot_state[1, 0], robot_radius)
    return collision_circle_polygon(robot_circle, polygon_vertices)

def collision_path_polygon(start_pos, end_pos, polygon_vertices):
    """
    检查路径（直线）与多边形是否相交
    Args:
        start_pos: [x, y] 或 np.array 起点
        end_pos: [x, y] 或 np.array 终点
        polygon_vertices: list of [x, y] 多边形顶点
    Returns:
        bool: 路径是否与多边形相交
    """
    # 处理numpy数组输入
    if hasattr(start_pos, 'shape'):
        if len(start_pos.shape) == 2:
            start_coord = [start_pos[0, 0], start_pos[1, 0]]
        else:
            start_coord = [start_pos[0], start_pos[1]]
    else:
        start_coord = start_pos
        
    if hasattr(end_pos, 'shape'):
        if len(end_pos.shape) == 2:
            end_coord = [end_pos[0, 0], end_pos[1, 0]]
        else:
            end_coord = [end_pos[0], end_pos[1]]
    else:
        end_coord = end_pos
    
    return PolygonGenerator.line_intersects_polygon(start_coord, end_coord, polygon_vertices)

def check_agent_safe_distance(agent_pos, agent_radius, polygon_vertices, safe_distance=0.5):
    """
    检查agent是否与多边形保持安全距离
    Args:
        agent_pos: [x, y] 或 np.array agent位置
        agent_radius: float agent半径
        polygon_vertices: list of [x, y] 多边形顶点
        safe_distance: float 额外的安全距离
    Returns:
        bool: 是否保持安全距离
    """
    # 处理输入格式
    if hasattr(agent_pos, 'shape'):
        if len(agent_pos.shape) == 2:
            # 2D数组 [[x], [y]]
            pos = [agent_pos[0, 0], agent_pos[1, 0]]
        else:
            # 1D数组 [x, y]
            pos = [agent_pos[0], agent_pos[1]]
    else:
        # 普通列表
        pos = agent_pos
    
    # 计算到多边形的距离
    distance = PolygonGenerator.distance_point_to_polygon(pos, polygon_vertices)
    
    # 检查是否保持安全距离
    return distance >= (agent_radius + safe_distance)

def check_path_safe_distance(start_pos, end_pos, polygon_vertices, path_width=0.4, safe_distance=0.3):
    """
    检查路径是否与多边形保持安全距离
    Args:
        start_pos: [x, y] 路径起点
        end_pos: [x, y] 路径终点
        polygon_vertices: list of [x, y] 多边形顶点
        path_width: float 路径宽度（机器人直径）
        safe_distance: float 额外安全距离
    Returns:
        bool: 路径是否安全
    """
    # 处理输入格式
    if hasattr(start_pos, 'shape'):
        if len(start_pos.shape) == 2:
            start = [start_pos[0, 0], start_pos[1, 0]]
        else:
            start = [start_pos[0], start_pos[1]]
    else:
        start = start_pos
        
    if hasattr(end_pos, 'shape'):
        if len(end_pos.shape) == 2:
            end = [end_pos[0, 0], end_pos[1, 0]]
        else:
            end = [end_pos[0], end_pos[1]]
    else:
        end = end_pos
    
    # 简化检查：沿路径采样点检查
    num_samples = 10
    for i in range(num_samples + 1):
        t = i / num_samples
        sample_x = start[0] + t * (end[0] - start[0])
        sample_y = start[1] + t * (end[1] - start[1])
        
        distance = PolygonGenerator.distance_point_to_polygon([sample_x, sample_y], polygon_vertices)
        
        if distance < (path_width / 2 + safe_distance):
            return False
    
    return True

def get_polygon_bounding_box(polygon_vertices):
    """
    获取多边形的边界框
    Args:
        polygon_vertices: list of [x, y] 多边形顶点
    Returns:
        tuple: (x_min, y_min, x_max, y_max)
    """
    if not polygon_vertices:
        return (0, 0, 0, 0)
    
    x_coords = [v[0] for v in polygon_vertices]
    y_coords = [v[1] for v in polygon_vertices]
    
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

def polygons_bounding_boxes_overlap(poly1_vertices, poly2_vertices):
    """
    检查两个多边形的边界框是否重叠（快速预检查）
    Args:
        poly1_vertices: list of [x, y] 第一个多边形顶点
        poly2_vertices: list of [x, y] 第二个多边形顶点
    Returns:
        bool: 边界框是否重叠
    """
    box1 = get_polygon_bounding_box(poly1_vertices)
    box2 = get_polygon_bounding_box(poly2_vertices)
    
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

def collision_circle_multiple_polygons(circle_obj, polygons_list):
    """
    检查圆与多个多边形的碰撞
    Args:
        circle_obj: circle namedtuple
        polygons_list: list of polygon_vertices 多边形列表
    Returns:
        bool: 是否与任何一个多边形碰撞
    """
    for polygon_vertices in polygons_list:
        if collision_circle_polygon(circle_obj, polygon_vertices):
            return True
    return False

def collision_line_multiple_polygons(line_start, line_end, polygons_list):
    """
    检查线段与多个多边形的碰撞
    Args:
        line_start: [x, y] 线段起点
        line_end: [x, y] 线段终点
        polygons_list: list of polygon_vertices 多边形列表
    Returns:
        bool: 是否与任何一个多边形碰撞
    """
    for polygon_vertices in polygons_list:
        if collision_line_polygon(line_start, line_end, polygon_vertices):
            return True
    return False

def get_safe_spawn_area(world_bounds, polygons_list, agent_radius, safe_distance=0.5):
    """
    获取可以安全生成agent的区域（避开所有多边形）
    Args:
        world_bounds: [x_min, y_min, x_max, y_max] 世界边界
        polygons_list: list of polygon_vertices 多边形列表
        agent_radius: float agent半径
        safe_distance: float 安全距离
    Returns:
        list: 安全区域的采样点列表 [[x, y], ...]
    """
    x_min, y_min, x_max, y_max = world_bounds
    safe_points = []
    
    # 网格采样
    step = 0.2
    x = x_min + agent_radius + safe_distance
    while x < x_max - agent_radius - safe_distance:
        y = y_min + agent_radius + safe_distance
        while y < y_max - agent_radius - safe_distance:
            point = [x, y]
            
            # 检查是否与所有多边形保持安全距离
            safe = True
            for polygon_vertices in polygons_list:
                if not check_agent_safe_distance(point, agent_radius, 
                                                polygon_vertices, safe_distance):
                    safe = False
                    break
            
            if safe:
                safe_points.append(point)
                
            y += step
        x += step
    
    return safe_points
