import numpy as np
from math import pi, cos, sin, sqrt
from collections import namedtuple
from ir_sim.world import obs_circle, obs_polygon
from ir_sim.util import collision_cir_cir, collision_cir_matrix, collision_cir_seg


class env_obs_random:
    """
    Random obstacle environment class that generates mixed types of static obstacles.
    Supports circles, polygons, and line segments with configurable parameters.
    """
    
    def __init__(self, obs_types=['circles'], obs_num_range=(2, 8), 
                 generation_area=[0, 0, 10, 10], obs_interval=1.0,
                 size_ranges=None, obstacle_density=0.15, regenerate_per_episode=True,
                 test_mode=False, test_scenarios=None, components=[], 
                 step_time=0.1, **kwargs):
        """
        Initialize random obstacle environment.
        
        Args:
            obs_types: List of obstacle types ['circles', 'polygons', 'lines']
            obs_num_range: Tuple (min_num, max_num) for obstacle count
            generation_area: [x_min, y_min, x_max, y_max] area for obstacle placement
            obs_interval: Minimum distance between obstacles
            size_ranges: Dict with size parameters for each obstacle type
            obstacle_density: Maximum fraction of area covered by obstacles
            regenerate_per_episode: Whether to regenerate obstacles each episode
            test_mode: Use predefined test scenarios
            test_scenarios: List of predefined scenario configurations
            components: Existing environment components for collision checking
        """
        
        self.obs_types = obs_types
        self.obs_num_range = obs_num_range
        self.generation_area = generation_area
        self.obs_interval = obs_interval
        self.obstacle_density = obstacle_density
        self.regenerate_per_episode = regenerate_per_episode
        self.test_mode = test_mode
        self.test_scenarios = test_scenarios or []
        self.components = components
        self.step_time = step_time
        
        # Set default size ranges if not provided
        self.size_ranges = size_ranges or {
            'circles': {'radius_range': [0.3, 0.8]},
            'polygons': {'size_range': [0.8, 1.5], 'vertex_num_range': [3, 6]},
            'lines': {'length_range': [1.0, 3.0]}
        }
        
        # Initialize obstacle lists
        self.obs_circle_list = []
        self.obs_polygon_list = []
        self.obs_line_states = []
        self.all_obstacles = []
        
        # Current scenario info
        self.current_scenario_id = 0
        self.total_area = self._calculate_area()
        
        # Generate initial obstacles
        self.regenerate_obstacles()
    
    def _calculate_area(self):
        """Calculate the total generation area."""
        return (self.generation_area[2] - self.generation_area[0]) * \
               (self.generation_area[3] - self.generation_area[1])
    
    def regenerate_obstacles(self):
        """Regenerate all obstacles based on current configuration."""
        # Clear existing obstacles
        self.obs_circle_list.clear()
        self.obs_polygon_list.clear()
        self.obs_line_states.clear()
        self.all_obstacles.clear()
        
        if self.test_mode and self.test_scenarios:
            self._generate_test_scenario()
        else:
            self._generate_random_obstacles()
    
    def _generate_test_scenario(self):
        """Generate obstacles from predefined test scenarios."""
        scenario = self.test_scenarios[self.current_scenario_id % len(self.test_scenarios)]
        
        if 'circles' in scenario.get('obs_types', []):
            positions = scenario.get('fixed_positions', [])
            sizes = scenario.get('fixed_sizes', [])
            
            for i, (pos, size) in enumerate(zip(positions, sizes)):
                state = np.array([[pos[0]], [pos[1]]])
                circle = obs_circle(id=i, state=state, radius=size, 
                                  obs_model='static', step_time=self.step_time)
                self.obs_circle_list.append(circle)
                self.all_obstacles.append(('circle', circle))
        
        # Add polygon and line generation for test scenarios if needed
        self.current_scenario_id += 1
    
    def _generate_random_obstacles(self):
        """Generate random obstacles with density and collision constraints."""
        # Determine number of obstacles
        obs_count = np.random.randint(self.obs_num_range[0], self.obs_num_range[1] + 1)
        
        # Calculate maximum allowed total area based on density
        max_obstacle_area = self.total_area * self.obstacle_density
        current_area = 0.0
        
        attempts = 0
        max_attempts = obs_count * 20  # Prevent infinite loops
        
        while len(self.all_obstacles) < obs_count and attempts < max_attempts:
            attempts += 1
            
            # Randomly select obstacle type
            obs_type = np.random.choice(self.obs_types)
            
            # Generate obstacle based on type
            if obs_type == 'circles':
                obstacle, area = self._generate_random_circle()
            elif obs_type == 'polygons':
                obstacle, area = self._generate_random_polygon()
            elif obs_type == 'lines':
                obstacle, area = self._generate_random_line()
            else:
                continue
            
            # Check area constraint
            if current_area + area > max_obstacle_area:
                continue
            
            # Check collision with existing obstacles
            if obstacle is not None and self._is_valid_placement(obstacle, obs_type):
                if obs_type == 'circles':
                    self.obs_circle_list.append(obstacle)
                elif obs_type == 'polygons':
                    self.obs_polygon_list.append(obstacle)
                elif obs_type == 'lines':
                    self.obs_line_states.append(obstacle)
                
                self.all_obstacles.append((obs_type, obstacle))
                current_area += area
    
    def _generate_random_circle(self):
        """Generate a random circle obstacle."""
        # Random radius
        radius_range = self.size_ranges['circles']['radius_range']
        radius = np.random.uniform(radius_range[0], radius_range[1])
        
        # Random position (with margin for radius)
        margin = radius + self.obs_interval
        x_min = self.generation_area[0] + margin
        x_max = self.generation_area[2] - margin
        y_min = self.generation_area[1] + margin
        y_max = self.generation_area[3] - margin
        
        if x_max <= x_min or y_max <= y_min:
            return None, 0.0
        
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        
        state = np.array([[x], [y]])
        circle = obs_circle(id=len(self.obs_circle_list), state=state, radius=radius,
                           obs_model='static', step_time=self.step_time)
        
        area = pi * radius * radius
        return circle, area
    
    def _generate_random_polygon(self):
        """Generate a random polygon obstacle."""
        size_range = self.size_ranges['polygons']['size_range']
        vertex_range = self.size_ranges['polygons']['vertex_num_range']
        
        # Random polygon parameters
        size = np.random.uniform(size_range[0], size_range[1])
        vertex_num = np.random.randint(vertex_range[0], vertex_range[1] + 1)
        
        # Generate regular polygon with some randomness
        center_margin = size + self.obs_interval
        x_min = self.generation_area[0] + center_margin
        x_max = self.generation_area[2] - center_margin
        y_min = self.generation_area[1] + center_margin
        y_max = self.generation_area[3] - center_margin
        
        if x_max <= x_min or y_max <= y_min:
            return None, 0.0
        
        center_x = np.random.uniform(x_min, x_max)
        center_y = np.random.uniform(y_min, y_max)
        
        # Generate vertices around center
        vertices = []
        angle_step = 2 * pi / vertex_num
        base_radius = size / 2
        
        for i in range(vertex_num):
            angle = i * angle_step
            # Add some randomness to radius
            radius = base_radius * np.random.uniform(0.7, 1.3)
            x = center_x + radius * cos(angle)
            y = center_y + radius * sin(angle)
            vertices.append([x, y])
        
        polygon = obs_polygon(vertex=vertices)
        
        # Approximate area calculation
        area = size * size
        return polygon, area
    
    def _generate_random_line(self):
        """Generate a random line segment obstacle."""
        length_range = self.size_ranges['lines']['length_range']
        length = np.random.uniform(length_range[0], length_range[1])
        
        # Random starting point
        margin = self.obs_interval
        x_min = self.generation_area[0] + margin
        x_max = self.generation_area[2] - margin
        y_min = self.generation_area[1] + margin
        y_max = self.generation_area[3] - margin
        
        if x_max <= x_min or y_max <= y_min:
            return None, 0.0
        
        x1 = np.random.uniform(x_min, x_max)
        y1 = np.random.uniform(y_min, y_max)
        
        # Random direction
        angle = np.random.uniform(0, 2 * pi)
        x2 = x1 + length * cos(angle)
        y2 = y1 + length * sin(angle)
        
        # Ensure end point is within bounds
        x2 = np.clip(x2, self.generation_area[0], self.generation_area[2])
        y2 = np.clip(y2, self.generation_area[1], self.generation_area[3])
        
        line_state = [x1, y1, x2, y2]
        
        # Area is negligible for lines
        area = 0.0
        return line_state, area
    
    def _is_valid_placement(self, obstacle, obs_type):
        """Check if obstacle placement is valid (no collisions)."""
        if obs_type == 'circles':
            return self._check_circle_collision(obstacle)
        elif obs_type == 'polygons':
            return self._check_polygon_collision(obstacle)
        elif obs_type == 'lines':
            return self._check_line_collision(obstacle)
        return False
    
    def _check_circle_collision(self, new_circle):
        """Check if new circle collides with existing obstacles."""
        circle = namedtuple('circle', 'x y r')
        point = namedtuple('point', 'x y')
        
        new_circle_geom = circle(new_circle.state[0, 0], new_circle.state[1, 0], 
                                new_circle.radius + self.obs_interval)
        
        # Check collision with map boundaries
        if hasattr(self, 'components') and 'map_matrix' in self.components:
            if collision_cir_matrix(new_circle_geom, self.components['map_matrix'], 
                                  self.components['xy_reso'], self.components['offset']):
                return False
        
        # Check collision with existing circles
        for existing_circle in self.obs_circle_list:
            existing_geom = circle(existing_circle.state[0, 0], existing_circle.state[1, 0],
                                 existing_circle.radius + self.obs_interval)
            if collision_cir_cir(new_circle_geom, existing_geom):
                return False
        
        # Check collision with existing line segments
        for line_state in self.obs_line_states:
            segment = [point(line_state[0], line_state[1]), 
                      point(line_state[2], line_state[3])]
            if collision_cir_seg(new_circle_geom, segment):
                return False
        
        # Check collision with existing polygons
        for polygon in self.obs_polygon_list:
            for edge in polygon.edge_list:
                segment = [point(edge[0], edge[1]), point(edge[2], edge[3])]
                if collision_cir_seg(new_circle_geom, segment):
                    return False
        
        return True
    
    def _check_polygon_collision(self, new_polygon):
        """Check if new polygon collides with existing obstacles."""
        # Simplified collision check - check if polygon center is far enough
        # from existing obstacles
        center_x = np.mean(new_polygon.vertexes[0, :])
        center_y = np.mean(new_polygon.vertexes[1, :])
        center_point = np.array([[center_x], [center_y]])
        
        # Check distance to existing circles
        for existing_circle in self.obs_circle_list:
            distance = np.linalg.norm(center_point - existing_circle.state)
            if distance < (existing_circle.radius + self.obs_interval):
                return False
        
        # Check with existing polygons (simplified)
        for existing_polygon in self.obs_polygon_list:
            existing_center_x = np.mean(existing_polygon.vertexes[0, :])
            existing_center_y = np.mean(existing_polygon.vertexes[1, :])
            distance = sqrt((center_x - existing_center_x)**2 + (center_y - existing_center_y)**2)
            if distance < self.obs_interval * 2:
                return False
        
        return True
    
    def _check_line_collision(self, new_line):
        """Check if new line collides with existing obstacles."""
        circle = namedtuple('circle', 'x y r')
        point = namedtuple('point', 'x y')
        
        # Create line segment
        line_segment = [point(new_line[0], new_line[1]), 
                       point(new_line[2], new_line[3])]
        
        # Check collision with existing circles
        for existing_circle in self.obs_circle_list:
            circle_geom = circle(existing_circle.state[0, 0], existing_circle.state[1, 0],
                               existing_circle.radius + self.obs_interval)
            if collision_cir_seg(circle_geom, line_segment):
                return False
        
        # Check minimum distance to existing lines
        for existing_line in self.obs_line_states:
            # Simplified distance check between line midpoints
            new_mid_x = (new_line[0] + new_line[2]) / 2
            new_mid_y = (new_line[1] + new_line[3]) / 2
            existing_mid_x = (existing_line[0] + existing_line[2]) / 2
            existing_mid_y = (existing_line[1] + existing_line[3]) / 2
            
            distance = sqrt((new_mid_x - existing_mid_x)**2 + (new_mid_y - existing_mid_y)**2)
            if distance < self.obs_interval:
                return False
        
        return True
    
    def get_obstacle_count(self):
        """Return the total number of generated obstacles."""
        return len(self.all_obstacles)
    
    def get_obstacle_density(self):
        """Calculate and return current obstacle density."""
        total_obstacle_area = 0.0
        
        # Calculate area for circles
        for circle in self.obs_circle_list:
            total_obstacle_area += pi * circle.radius * circle.radius
        
        # Approximate area for polygons
        for polygon in self.obs_polygon_list:
            # Use bounding box as approximation
            x_min, x_max = np.min(polygon.vertexes[0, :]), np.max(polygon.vertexes[0, :])
            y_min, y_max = np.min(polygon.vertexes[1, :]), np.max(polygon.vertexes[1, :])
            total_obstacle_area += (x_max - x_min) * (y_max - y_min)
        
        return total_obstacle_area / self.total_area
    
    def reset(self, **kwargs):
        """Reset the environment, potentially regenerating obstacles."""
        if self.regenerate_per_episode:
            self.regenerate_obstacles()
    
    def get_all_obstacles_info(self):
        """Return information about all generated obstacles for debugging/visualization."""
        info = {
            'circles': [{'center': [c.state[0, 0], c.state[1, 0]], 'radius': c.radius} 
                       for c in self.obs_circle_list],
            'polygons': [{'vertices': p.vertexes.T.tolist()} for p in self.obs_polygon_list],
            'lines': [{'start': [l[0], l[1]], 'end': [l[2], l[3]]} for l in self.obs_line_states],
            'total_count': self.get_obstacle_count(),
            'density': self.get_obstacle_density()
        }
        return info
