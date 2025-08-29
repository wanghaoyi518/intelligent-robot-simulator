import numpy as np
from math import pi, cos, sin
from ir_sim.world import obs_polygon
from ir_sim.env.env_obs_poly import env_obs_poly
from ir_sim.util import PolygonGenerator, collision_circle_polygon, check_agent_safe_distance
import random

class env_obs_poly_random(env_obs_poly):
    """随机多边形障碍物环境管理器"""
    
    def __init__(self, obs_poly_class=obs_polygon, components=[], **kwargs):
        # 随机生成参数
        self.obs_num_range = kwargs.get('obs_num_range', [2, 5])
        self.polygon_size_range = kwargs.get('polygon_size_range', [0.8, 1.5])
        self.vertex_num_range = kwargs.get('vertex_num_range', [3, 6])
        self.generation_area = kwargs.get('generation_area', [1, 1, 9, 9])
        self.min_obstacle_distance = kwargs.get('min_obstacle_distance', 1.0)
        self.min_agent_distance = kwargs.get('min_agent_distance', 0.5)
        self.obstacle_density = kwargs.get('obstacle_density', 0.1)
        self.irregularity = kwargs.get('irregularity', 0.3)
        self.max_attempts = kwargs.get('max_attempts', 100)
        
        # 世界参数
        self.world_bounds = kwargs.get('world_bounds', [0, 0, 10, 10])
        
        # 组件引用（用于与机器人等其他组件交互）
        self.components = components
        
        # 生成随机多边形
        vertex_list = self.generate_random_polygon_vertices()
        obs_poly_num = len(vertex_list)
        
        print(f"🔶 Mode 7: 生成了 {obs_poly_num} 个随机多边形障碍物")
        
        # 调用父类初始化
        super().__init__(obs_poly_class=obs_poly_class, 
                        vertex_list=vertex_list, 
                        obs_poly_num=obs_poly_num, 
                        **kwargs)
    
    def generate_random_polygon_vertices(self):
        """
        生成随机多边形顶点列表
        Returns:
            list: 多边形顶点列表 [[[x1,y1], [x2,y2], ...], ...]
        """
        # 随机确定多边形数量
        num_polygons = random.randint(self.obs_num_range[0], self.obs_num_range[1])
        
        vertex_list = []
        existing_polygons = []
        
        # 检查是否超过密度限制
        total_area = self.calculate_generation_area()
        max_total_polygon_area = total_area * self.obstacle_density
        current_total_area = 0
        
        for i in range(num_polygons):
            # 尝试生成一个新的多边形
            for attempt in range(self.max_attempts):
                # 生成新多边形
                new_polygon = PolygonGenerator.generate_random_polygon_in_area(
                    self.generation_area,
                    self.polygon_size_range,
                    self.vertex_num_range,
                    existing_polygons,
                    self.min_obstacle_distance,
                    max_attempts=50
                )
                
                if new_polygon is None:
                    continue
                
                # 检查面积限制
                polygon_area = PolygonGenerator.polygon_area(new_polygon)
                if current_total_area + polygon_area > max_total_polygon_area:
                    continue
                
                # 成功生成多边形
                vertex_list.append(new_polygon)
                existing_polygons.append(new_polygon)
                current_total_area += polygon_area
                
                print(f"  ✅ 多边形 {i+1}: {len(new_polygon)} 个顶点, 面积: {polygon_area:.2f}")
                break
            else:
                print(f"  ⚠️  多边形 {i+1}: 生成失败，跳过")
        
        return vertex_list
    
    def calculate_generation_area(self):
        """计算生成区域的面积"""
        x_min, y_min, x_max, y_max = self.generation_area
        return (x_max - x_min) * (y_max - y_min)
    
    def regenerate_polygons(self, **kwargs):
        """
        重新生成多边形障碍物（用于reset时）
        Args:
            **kwargs: 可以传入新的生成参数
        """
        # 更新参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # 重新生成
        vertex_list = self.generate_random_polygon_vertices()
        obs_poly_num = len(vertex_list)
        
        # 重新创建多边形对象列表
        self.obs_poly_list = [
            obs_polygon(vertex=v, **kwargs) 
            for v in vertex_list[0:obs_poly_num]
        ]
        
        print(f"🔄 重新生成了 {obs_poly_num} 个多边形障碍物")
    
    def check_collision_with_agents(self, agent_positions, agent_radii):
        """
        检查多边形是否与agent位置冲突
        Args:
            agent_positions: list of [x, y] agent位置列表
            agent_radii: list of float agent半径列表
        Returns:
            bool: 是否存在冲突
        """
        for i, pos in enumerate(agent_positions):
            radius = agent_radii[i] if i < len(agent_radii) else 0.2
            
            for polygon in self.obs_poly_list:
                if not check_agent_safe_distance(pos, radius, 
                                                polygon.vertexes.T.tolist(), 
                                                self.min_agent_distance):
                    return True
        return False
    
    def check_path_collision(self, start_pos, end_pos, path_width=0.4):
        """
        检查路径是否与任何多边形碰撞
        Args:
            start_pos: [x, y] 起点
            end_pos: [x, y] 终点
            path_width: float 路径宽度
        Returns:
            bool: 是否碰撞
        """
        from ir_sim.util import check_path_safe_distance
        
        for polygon in self.obs_poly_list:
            if not check_path_safe_distance(start_pos, end_pos,
                                          polygon.vertexes.T.tolist(),
                                          path_width, safe_distance=0.1):
                return True
        return False
    
    def get_safe_spawn_points(self, num_points, agent_radius=0.2, safety_margin=0.5):
        """
        获取可以安全生成agent的位置点
        Args:
            num_points: int 需要的点数量
            agent_radius: float agent半径
            safety_margin: float 安全边距
        Returns:
            list: 安全位置列表 [[x, y], ...]
        """
        from ir_sim.util import get_safe_spawn_area
        
        # 获取所有多边形的顶点列表
        polygons_list = [polygon.vertexes.T.tolist() for polygon in self.obs_poly_list]
        
        # 获取安全区域
        safe_points = get_safe_spawn_area(
            self.world_bounds, 
            polygons_list, 
            agent_radius, 
            safety_margin
        )
        
        # 随机选择所需数量的点
        if len(safe_points) < num_points:
            print(f"⚠️  只找到 {len(safe_points)} 个安全点，需要 {num_points} 个")
            return safe_points
        
        selected_points = random.sample(safe_points, num_points)
        return selected_points
    
    def get_polygon_info(self):
        """
        获取多边形信息用于调试和可视化
        Returns:
            dict: 多边形信息
        """
        info = {
            'count': len(self.obs_poly_list),
            'polygons': []
        }
        
        for i, polygon in enumerate(self.obs_poly_list):
            vertices = polygon.vertexes.T.tolist()
            center = PolygonGenerator.get_polygon_center(vertices)
            area = PolygonGenerator.polygon_area(vertices)
            
            poly_info = {
                'id': i,
                'vertices': vertices,
                'center': center,
                'area': area,
                'vertex_count': len(vertices)
            }
            info['polygons'].append(poly_info)
        
        return info
    
    def validate_configuration(self):
        """
        验证配置参数的合理性
        Returns:
            bool: 配置是否合理
        """
        warnings = []
        
        # 检查生成区域
        x_min, y_min, x_max, y_max = self.generation_area
        if x_max - x_min < 2 or y_max - y_min < 2:
            warnings.append("生成区域过小")
        
        # 检查多边形数量与密度
        area = (x_max - x_min) * (y_max - y_min)
        max_polygons = int(area * self.obstacle_density / 0.5)  # 假设平均面积0.5
        if self.obs_num_range[1] > max_polygons:
            warnings.append(f"多边形数量过多，建议最大值不超过 {max_polygons}")
        
        # 检查最小距离
        if self.min_obstacle_distance < 0.5:
            warnings.append("障碍物间距过小，可能导致生成失败")
        
        if warnings:
            print("⚠️  配置警告:")
            for warning in warnings:
                print(f"   - {warning}")
            return False
        
        return True
    
    def reset(self, **kwargs):
        """
        重置多边形障碍物（用于episode重置）
        Args:
            **kwargs: 可选的新参数
        """
        regenerate = kwargs.get('regenerate_polygons', True)
        
        if regenerate:
            self.regenerate_polygons(**kwargs)
        
        # 可以在这里添加其他重置逻辑
    
    def step(self, **kwargs):
        """
        多边形障碍物的step操作（当前为静态，无需操作）
        """
        # 静态障碍物无需step操作
        pass
