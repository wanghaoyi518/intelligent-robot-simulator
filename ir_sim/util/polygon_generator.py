import numpy as np
from math import pi, cos, sin, sqrt, atan2
import random

class PolygonGenerator:
    """随机多边形生成工具类"""
    
    @staticmethod
    def generate_regular_polygon(center, size, vertex_num):
        """
        生成规则多边形
        Args:
            center: [x, y] 中心点坐标
            size: float 多边形大小（外接圆半径）
            vertex_num: int 顶点数量
        Returns:
            vertices: list of [x, y] 顶点坐标列表
        """
        vertices = []
        angle_step = 2 * pi / vertex_num
        
        for i in range(vertex_num):
            angle = i * angle_step
            x = center[0] + size * cos(angle)
            y = center[1] + size * sin(angle)
            vertices.append([x, y])
            
        return vertices
    
    @staticmethod
    def generate_irregular_polygon(center, size, vertex_num, irregularity=0.3):
        """
        生成不规则多边形
        Args:
            center: [x, y] 中心点坐标
            size: float 多边形大小（平均半径）
            vertex_num: int 顶点数量
            irregularity: float 不规则程度 (0-1, 0为完全规则)
        Returns:
            vertices: list of [x, y] 顶点坐标列表
        """
        vertices = []
        angle_step = 2 * pi / vertex_num
        
        for i in range(vertex_num):
            # 角度随机偏移
            angle = i * angle_step + random.uniform(-irregularity * angle_step/2, 
                                                   irregularity * angle_step/2)
            
            # 半径随机偏移
            radius = size + random.uniform(-irregularity * size, irregularity * size)
            radius = max(radius, size * 0.3)  # 确保最小半径
            
            x = center[0] + radius * cos(angle)
            y = center[1] + radius * sin(angle)
            vertices.append([x, y])
            
        return vertices
    
    @staticmethod
    def generate_random_polygon_in_area(generation_area, size_range, vertex_num_range, 
                                      existing_polygons=[], min_distance=1.0, 
                                      max_attempts=100):
        """
        在指定区域内生成随机多边形，避免与已有多边形重叠
        Args:
            generation_area: [x_min, y_min, x_max, y_max] 生成区域
            size_range: [min_size, max_size] 尺寸范围
            vertex_num_range: [min_vertices, max_vertices] 顶点数范围
            existing_polygons: list 已存在的多边形顶点列表
            min_distance: float 最小距离
            max_attempts: int 最大尝试次数
        Returns:
            vertices: list of [x, y] 或 None (如果生成失败)
        """
        x_min, y_min, x_max, y_max = generation_area
        
        for attempt in range(max_attempts):
            # 随机生成中心点
            center_x = random.uniform(x_min, x_max)
            center_y = random.uniform(y_min, y_max)
            center = [center_x, center_y]
            
            # 随机生成参数
            size = random.uniform(size_range[0], size_range[1])
            vertex_num = random.randint(vertex_num_range[0], vertex_num_range[1])
            irregularity = random.uniform(0.1, 0.4)
            
            # 生成多边形
            vertices = PolygonGenerator.generate_irregular_polygon(
                center, size, vertex_num, irregularity)
            
            # 检查是否在边界内
            if not PolygonGenerator.polygon_in_bounds(vertices, generation_area):
                continue
                
            # 检查是否与已有多边形重叠
            valid = True
            for existing_poly in existing_polygons:
                if PolygonGenerator.check_polygon_overlap(vertices, existing_poly, min_distance):
                    valid = False
                    break
                    
            if valid:
                return vertices
                
        return None  # 生成失败
    
    @staticmethod
    def polygon_in_bounds(vertices, bounds):
        """
        检查多边形是否完全在边界内
        Args:
            vertices: list of [x, y] 多边形顶点
            bounds: [x_min, y_min, x_max, y_max] 边界
        Returns:
            bool: 是否在边界内
        """
        x_min, y_min, x_max, y_max = bounds
        
        for vertex in vertices:
            x, y = vertex
            if x < x_min or x > x_max or y < y_min or y > y_max:
                return False
                
        return True
    
    @staticmethod
    def check_polygon_overlap(poly1_vertices, poly2_vertices, min_distance):
        """
        检查两个多边形是否重叠或距离过近
        Args:
            poly1_vertices: list of [x, y] 第一个多边形顶点
            poly2_vertices: list of [x, y] 第二个多边形顶点  
            min_distance: float 最小允许距离
        Returns:
            bool: True表示重叠或距离过近
        """
        # 简化算法：检查中心点距离
        center1 = PolygonGenerator.get_polygon_center(poly1_vertices)
        center2 = PolygonGenerator.get_polygon_center(poly2_vertices)
        
        distance = sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # 估算多边形半径
        radius1 = PolygonGenerator.get_polygon_radius(poly1_vertices)
        radius2 = PolygonGenerator.get_polygon_radius(poly2_vertices)
        
        # 检查是否距离过近
        return distance < (radius1 + radius2 + min_distance)
    
    @staticmethod
    def get_polygon_center(vertices):
        """计算多边形中心点"""
        x_sum = sum(vertex[0] for vertex in vertices)
        y_sum = sum(vertex[1] for vertex in vertices)
        return [x_sum / len(vertices), y_sum / len(vertices)]
    
    @staticmethod
    def get_polygon_radius(vertices):
        """计算多边形的近似半径（中心到最远顶点的距离）"""
        center = PolygonGenerator.get_polygon_center(vertices)
        max_distance = 0
        
        for vertex in vertices:
            distance = sqrt((vertex[0] - center[0])**2 + (vertex[1] - center[1])**2)
            max_distance = max(max_distance, distance)
            
        return max_distance
    
    @staticmethod
    def point_in_polygon(point, polygon_vertices):
        """
        检查点是否在多边形内（射线投射算法）
        Args:
            point: [x, y] 检测点
            polygon_vertices: list of [x, y] 多边形顶点
        Returns:
            bool: 点是否在多边形内
        """
        x, y = point
        n = len(polygon_vertices)
        inside = False
        
        p1x, p1y = polygon_vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    
    @staticmethod
    def line_intersects_polygon(line_start, line_end, polygon_vertices):
        """
        检查线段是否与多边形相交
        Args:
            line_start: [x, y] 线段起点
            line_end: [x, y] 线段终点  
            polygon_vertices: list of [x, y] 多边形顶点
        Returns:
            bool: 是否相交
        """
        # 检查线段端点是否在多边形内
        if (PolygonGenerator.point_in_polygon(line_start, polygon_vertices) or
            PolygonGenerator.point_in_polygon(line_end, polygon_vertices)):
            return True
            
        # 检查线段是否与多边形边界相交
        n = len(polygon_vertices)
        for i in range(n):
            edge_start = polygon_vertices[i]
            edge_end = polygon_vertices[(i + 1) % n]
            
            if PolygonGenerator.line_segments_intersect(
                line_start, line_end, edge_start, edge_end):
                return True
                
        return False
    
    @staticmethod
    def line_segments_intersect(p1, p2, p3, p4):
        """
        检查两个线段是否相交
        Args:
            p1, p2: [x, y] 第一条线段的端点
            p3, p4: [x, y] 第二条线段的端点
        Returns:
            bool: 是否相交
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    @staticmethod
    def polygon_area(vertices):
        """计算多边形面积（鞋带公式）"""
        n = len(vertices)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
            
        return abs(area) / 2.0
    
    @staticmethod
    def distance_point_to_polygon(point, polygon_vertices):
        """计算点到多边形的最小距离"""
        min_distance = float('inf')
        
        # 如果点在多边形内，距离为0
        if PolygonGenerator.point_in_polygon(point, polygon_vertices):
            return 0.0
            
        # 计算点到各边的距离
        n = len(polygon_vertices)
        for i in range(n):
            edge_start = polygon_vertices[i]
            edge_end = polygon_vertices[(i + 1) % n]
            
            distance = PolygonGenerator.distance_point_to_line_segment(
                point, edge_start, edge_end)
            min_distance = min(min_distance, distance)
            
        return min_distance
    
    @staticmethod
    def distance_point_to_line_segment(point, line_start, line_end):
        """计算点到线段的最小距离"""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:  # 线段退化为点
            return sqrt(A * A + B * B)
            
        param = dot / len_sq
        
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx, yy = x1 + param * C, y1 + param * D
            
        dx = px - xx
        dy = py - yy
        
        return sqrt(dx * dx + dy * dy)
