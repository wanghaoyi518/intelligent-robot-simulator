from ir_sim.world import mobile_robot
from math import pi, cos, sin
import numpy as np
from collections import namedtuple
from ir_sim.util import collision_cir_cir, collision_cir_matrix, collision_cir_seg

class env_robot:
    def __init__(self, robot_class=mobile_robot, robot_number=0, robot_mode='omni', robot_init_mode = 0, step_time=0.1, components=[], **kwargs):

        self.robot_class = robot_class
        self.robot_number = robot_number
        self.init_mode = robot_init_mode
        self.robot_list = []
        self.cur_mode = robot_init_mode
        self.com = components

        self.interval = kwargs.get('interval', 1)
        self.square = kwargs.get('square', [0, 0, 10, 10] )
        self.circular = kwargs.get('circular', [5, 5, 4] )
        self.random_bear = kwargs.get('random_bear', False)
        self.random_radius = kwargs.get('random_radius', False)
        self.max_start_goal_distance = kwargs.get('max_start_goal_distance', 5.0)


        # init_mode: 0 manually initialize
        #            1 single row
        #            2 random
        #            3 circular 
        #            4 random 2
        #            5 corridor
        #            6 random with distance constraint
        # kwargs: random_bear random radius
        if self.robot_number > 0:
            if self.init_mode == 0:
                assert 'radius_list' and 'init_state_list' and 'goal_list' in kwargs.keys()
                radius_list = kwargs['radius_list']
                init_state_list = kwargs['init_state_list']
                goal_list = kwargs['goal_list']
            else:
                radius_list = kwargs.get('radius_list', [0.2])
                init_state_list, goal_list, radius_list = self.init_state_distribute(self.init_mode, radius=radius_list[0])

        # robot
        for i in range(self.robot_number):
            robot = self.robot_class(id=i, mode=robot_mode, radius=radius_list[i], init_state=init_state_list[i], goal=goal_list[i], step_time=step_time, **kwargs)
            self.robot_list.append(robot)
            self.robot = robot if i == 0 else None 
        
    def init_state_distribute(self, init_mode=1, radius=0.2):
        # init_mode: 1 single row
        #            2 random
        #            3 circular 
        #            4 random 2
        #            5 corridor
        #            6 random with distance constraint
        #            7 random with distance constraint + random polygons
        # square area: x_min, y_min, x_max, y_max
        # circular area: x, y, radius
        
        num = self.robot_number
        state_list, goal_list = [], []

        if init_mode == 1:
             # single row
            state_list = [np.array([ [i * self.interval], [self.square[1]], [pi/2] ]) for i in range(int(self.square[0]), int(self.square[0])+num)]
            goal_list = [np.array([ [i * self.interval], [self.square[3]] ]) for i in range(int(self.square[0]), int(self.square[0])+num)]
            goal_list.reverse()

        elif init_mode == 2:
            # random
            state_list, goal_list = self.random_start_goal()

        elif init_mode == 3:
            # circular
            circle_point = np.array(self.circular)
            theta_step = 2*pi / num
            theta = 0

            while theta < 2*pi:
                state = circle_point + np.array([ cos(theta) * self.circular[2], sin(theta) * self.circular[2], theta + pi- self.circular[2] ])
                goal = circle_point[0:2] + np.array([cos(theta+pi), sin(theta+pi)]) * self.circular[2]
                theta = theta + theta_step
                state_list.append(state[:, np.newaxis])
                goal_list.append(goal[:, np.newaxis])

        elif init_mode == 4:
            # random 2
            circle_point = np.array(self.circular)
            theta_step = 2*pi / num
            theta = 0

            while theta < 2*pi:
                state = circle_point + np.array([ cos(theta) * self.circular[2], sin(theta) * self.circular[2], theta + pi- self.circular[2] ])
                goal = circle_point[0:2] + np.array([cos(theta+pi), sin(theta+pi)]) * self.circular[2]
                theta = theta + theta_step
                state_list.append(state[:, np.newaxis])
                goal_list.append(goal[:, np.newaxis])

        elif init_mode == 5:
            
            half_num = int(num /2)

            state_list1 = [np.array([ [i * self.interval], [self.square[1]], [pi/2] ]) for i in range(int(self.square[0]), int(self.square[0])+half_num)]

            state_list2 = [np.array([ [i * self.interval], [self.square[3]], [pi/2] ]) for i in range(int(self.square[0]), int(self.square[0])+half_num)]
            state_list2.reverse()
            
            goal_list1 = [np.array([ [i * self.interval], [self.square[3]], [pi/2] ]) for i in range(int(self.square[0]), int(self.square[0])+half_num)]
            goal_list1.reverse()

            goal_list2 = [np.array([ [i * self.interval], [self.square[1]], [pi/2] ]) for i in range(int(self.square[0]), int(self.square[0])+half_num)]
            
            state_list, goal_list = state_list1+state_list2, goal_list1+goal_list2
        
        elif init_mode == 6:
            # random with distance constraint
            state_list, goal_list = self.random_start_goal_constrained(max_distance=self.max_start_goal_distance)
        
        elif init_mode == 7:
            # Mode 7: random with distance constraint + random polygons
            state_list, goal_list = self.random_start_goal_with_polygons(max_distance=self.max_start_goal_distance)
                    
        if self.random_bear:
            for state in state_list:
                state[2, 0] = np.random.uniform(low = -pi, high = pi)

        if self.random_radius:
            radius_list = np.random.uniform(low = 0.2, high = 1, size = (num,))
        else:
            radius_list = [radius for i in range(num)]

        return state_list, goal_list, radius_list
    
    def random_start_goal(self):

        num = self.robot_number
        random_list = []
        goal_list = []
        while len(random_list) < 2*num:

            new_point = np.random.uniform(low = self.square[0:2]+[-pi], high = self.square[2:4]+[pi], size = (1, 3)).T

            if not self.check_collision(new_point, random_list, self.com, self.interval):
                random_list.append(new_point)

        start_list = random_list[0 : num]
        goal_temp_list = random_list[num : 2 * num]

        for goal in goal_temp_list:
            goal_list.append(np.delete(goal, 2, 0))

        return start_list, goal_list
    
    def random_goal(self):

        num = self.robot_number
        random_list = []
        goal_list = []
        while len(random_list) < num:

            new_point = np.random.uniform(low = self.square[0:2]+[-pi], high = self.square[2:4]+[pi], size = (1, 3)).T

            if not self.check_collision(new_point, random_list, self.com, self.interval):
                random_list.append(new_point)

        goal_temp_list = random_list[:]

        for goal in goal_temp_list:
            goal_list.append(np.delete(goal, 2, 0))

        return goal_list

    def random_start_goal_constrained(self, max_distance=5.0):
        """
        Generate random start and goal points with distance constraint.
        Each agent's goal point must be within max_distance from its start point.
        """
        num = self.robot_number
        start_list = []
        goal_list = []
        
        # First, generate all valid start points
        while len(start_list) < num:
            new_start = np.random.uniform(low=self.square[0:2]+[-pi], high=self.square[2:4]+[pi], size=(1, 3)).T
            
            if not self.check_collision(new_start, start_list, self.com, self.interval):
                start_list.append(new_start)
        
        # Then, for each start point, generate a goal point within max_distance
        for start_point in start_list:
            max_attempts = 1000  # Prevent infinite loop
            attempts = 0
            goal_found = False
            
            while not goal_found and attempts < max_attempts:
                # Generate random angle and distance
                angle = np.random.uniform(0, 2*pi)
                distance = np.random.uniform(0, max_distance)
                
                # Calculate goal position relative to start
                goal_x = start_point[0, 0] + distance * cos(angle)
                goal_y = start_point[1, 0] + distance * sin(angle)
                
                # Check if goal is within the environment boundaries
                if (self.square[0] <= goal_x <= self.square[2] and 
                    self.square[1] <= goal_y <= self.square[3]):
                    
                    goal_point = np.array([[goal_x], [goal_y], [0]])  # Add dummy angle for collision check
                    
                    # Check collision for goal point (only against obstacles, not other goals)
                    if not self.check_collision(goal_point, [], self.com, self.interval/2):
                        goal_list.append(np.array([[goal_x], [goal_y]]))  # Remove angle for final goal
                        goal_found = True
                
                attempts += 1
            
            # Fallback: if no valid goal found within max_distance, use the start point as goal
            if not goal_found:
                print(f"Warning: Could not find valid goal within distance {max_distance} for start point {start_point[0:2].T}, using start as goal.")
                goal_list.append(start_point[0:2])  # Use start position as goal (no movement)
        
        return start_list, goal_list
    
    def random_start_goal_with_polygons(self, max_distance=5.0):
        """
        Mode 7: Generate random start and goal points with distance constraint,
        avoiding random polygon obstacles.
        """
        num = self.robot_number
        start_list = []
        goal_list = []
        
        # 获取多边形障碍物信息
        polygon_env = self.com.get('obs_polygons', None)
        if polygon_env is None or not hasattr(polygon_env, 'obs_poly_list'):
            print("⚠️  Mode 7: 未找到多边形障碍物，退回到Mode 6行为")
            return self.random_start_goal_constrained(max_distance)
        
        polygons_list = [poly.vertexes.T.tolist() for poly in polygon_env.obs_poly_list]
        
        # 获取安全的生成点
        safe_radius = 0.2  # 默认机器人半径
        safe_margin = 0.3  # 安全边距
        
        print(f"🔶 Mode 7: 在{len(polygons_list)}个多边形障碍物中生成{num}个机器人位置")
        
        # 生成起点
        max_attempts = 2000
        attempts = 0
        
        while len(start_list) < num and attempts < max_attempts:
            # 随机生成候选起点
            new_start = np.random.uniform(
                low=self.square[0:2]+[-pi], 
                high=self.square[2:4]+[pi], 
                size=(1, 3)
            ).T
            
            start_pos = new_start[0:2].flatten()
            
            # 检查与其他机器人的碰撞
            collision_with_robots = self.check_collision(new_start, start_list, self.com, self.interval)
            
            # 检查与多边形障碍物的碰撞
            collision_with_polygons = False
            if polygons_list:
                from ir_sim.util import check_agent_safe_distance
                for polygon_vertices in polygons_list:
                    if not check_agent_safe_distance(start_pos, safe_radius, polygon_vertices, safe_margin):
                        collision_with_polygons = True
                        break
            
            if not collision_with_robots and not collision_with_polygons:
                start_list.append(new_start)
                print(f"  ✅ 起点 {len(start_list)}: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
            
            attempts += 1
        
        if len(start_list) < num:
            print(f"⚠️  只生成了{len(start_list)}个起点，需要{num}个")
        
        # 为每个起点生成终点
        for i, start_point in enumerate(start_list):
            max_goal_attempts = 1000
            goal_attempts = 0
            goal_found = False
            
            while not goal_found and goal_attempts < max_goal_attempts:
                # 在最大距离内随机生成终点
                angle = np.random.uniform(0, 2*pi)
                distance = np.random.uniform(1.0, max_distance)  # 最小距离1.0避免起点终点太近
                
                # 计算终点位置
                goal_x = start_point[0, 0] + distance * cos(angle)
                goal_y = start_point[1, 0] + distance * sin(angle)
                
                # 检查是否在环境边界内
                if not (self.square[0] <= goal_x <= self.square[2] and 
                       self.square[1] <= goal_y <= self.square[3]):
                    goal_attempts += 1
                    continue
                
                goal_pos = [goal_x, goal_y]
                
                # 检查与多边形障碍物的碰撞
                collision_with_polygons = False
                if polygons_list:
                    from ir_sim.util import check_agent_safe_distance, check_path_safe_distance
                    
                    # 检查终点是否安全
                    for polygon_vertices in polygons_list:
                        if not check_agent_safe_distance(goal_pos, safe_radius, polygon_vertices, safe_margin):
                            collision_with_polygons = True
                            break
                    
                    # 检查路径是否安全
                    if not collision_with_polygons:
                        start_pos = start_point[0:2].flatten()
                        for polygon_vertices in polygons_list:
                            if not check_path_safe_distance(start_pos, goal_pos, polygon_vertices, 
                                                          path_width=safe_radius*2, safe_distance=0.2):
                                collision_with_polygons = True
                                break
                
                # 检查与其他障碍物的碰撞（使用原有的check_collision）
                goal_point_3d = np.array([[goal_x], [goal_y], [0]])
                collision_with_other = self.check_collision(goal_point_3d, [], self.com, self.interval/2)
                
                if not collision_with_polygons and not collision_with_other:
                    goal_list.append(np.array([[goal_x], [goal_y]]))
                    goal_found = True
                    actual_distance = np.sqrt((goal_x - start_point[0, 0])**2 + (goal_y - start_point[1, 0])**2)
                    print(f"  ✅ 终点 {i+1}: ({goal_x:.2f}, {goal_y:.2f}) 距离: {actual_distance:.2f}")
                
                goal_attempts += 1
            
            # 回退策略：如果无法找到有效终点
            if not goal_found:
                print(f"⚠️  机器人 {i+1}: 无法找到有效终点，使用起点作为终点")
                goal_list.append(start_point[0:2])
        
        print(f"🎯 Mode 7: 成功生成 {len(start_list)} 个起点和 {len(goal_list)} 个终点")
        
        return start_list, goal_list

    def distance(self, point1, point2):
        diff = point2[0:2] - point1[0:2]
        return np.linalg.norm(diff)

    def check_collision(self, check_point, point_list, components, range):

        circle = namedtuple('circle', 'x y r')
        point = namedtuple('point', 'x y')
        self_circle = circle(check_point[0, 0], check_point[1, 0], range/2)

        for obs_cir in components['obs_circles'].obs_cir_list:
            temp_circle = circle(obs_cir.state[0, 0], obs_cir.state[1, 0], obs_cir.radius)
            if collision_cir_cir(self_circle, temp_circle):
                return True
        
        # check collision with map
        if collision_cir_matrix(self_circle, components['map_matrix'], components['xy_reso'], components['offset']):
            return True

        # check collision with line obstacles
        for line in components['obs_lines'].obs_line_states:
            segment = [point(line[0], line[1]), point(line[2], line[3])]
            if collision_cir_seg(self_circle, segment):
                return True

        for point in point_list:
            if self.distance(check_point, point) < range:
                return True
                
        return False


    def step(self, vel_list=[], **vel_kwargs):

        # vel_kwargs: vel_type = 'diff', 'omni'
        #             stop=True, whether stop when arrive at the goal
        #             noise=False, 
        #             alpha = [0.01, 0, 0, 0.01, 0, 0], noise for diff
        #             control_std = [0.01, 0.01], noise for omni

        for robot, vel in zip(self.robot_list, vel_list):
            robot.move_forward(vel, **vel_kwargs)

    def cal_des_list(self):
        vel_list = list(map(lambda x: x.cal_des_vel() , self.robot_list))
        return vel_list
    
    def cal_des_omni_list(self):
        vel_list = list(map(lambda x: x.cal_des_vel_omni() , self.robot_list))
        return vel_list

    def arrive_all(self):

        for robot in self.robot_list:
            if not robot.arrive():
                return False

        return True

    def robots_reset(self, reset_mode=1, **kwargs):
        
        if reset_mode == 0:
            for robot in self.robot_list:
                robot.reset(self.random_bear)
        
        elif self.cur_mode != reset_mode:
            state_list, goal_list, _ = self.init_state_distribute(init_mode = reset_mode)

            for i in range(self.robot_number):
                self.robot_list[i].init_state = state_list[i]
                self.robot_list[i].goal = goal_list[i]
                self.robot_list[i].reset(self.random_bear) 
            
            self.cur_mode = reset_mode

        elif reset_mode == 2:
            state_list, goal_list = self.random_start_goal()
            for i in range(self.robot_number):
                self.robot_list[i].init_state = state_list[i]
                self.robot_list[i].goal = goal_list[i]
                self.robot_list[i].reset(self.random_bear) 
        
        elif reset_mode == 4:
            goal_list = self.random_goal()
            for i in range(self.robot_number):
                self.robot_list[i].goal = goal_list[i]
                self.robot_list[i].reset(self.random_bear)
        
        elif reset_mode == 6:
            state_list, goal_list = self.random_start_goal_constrained(max_distance=self.max_start_goal_distance)
            for i in range(self.robot_number):
                self.robot_list[i].init_state = state_list[i]
                self.robot_list[i].goal = goal_list[i]
                self.robot_list[i].reset(self.random_bear)
        
        elif reset_mode == 7:
            # Mode 7: random with distance constraint + random polygons
            state_list, goal_list = self.random_start_goal_with_polygons(max_distance=self.max_start_goal_distance)
            for i in range(self.robot_number):
                self.robot_list[i].init_state = state_list[i]
                self.robot_list[i].goal = goal_list[i]
                self.robot_list[i].reset(self.random_bear)

        else:
            for robot in self.robot_list:
                robot.reset(self.random_bear)

    def robot_reset(self, id=0):
        self.robot_list[id].reset(self.random_bear)

    def total_states(self):
        robot_state_list = list(map(lambda r: np.squeeze( r.omni_state()), self.robot_list))
        nei_state_list = list(map(lambda r: np.squeeze( r.omni_obs_state()), self.robot_list))
        obs_circular_list = list(map(lambda o: np.squeeze( o.omni_obs_state() ), self.com['obs_circles'].obs_cir_list))
        obs_line_list = self.com['obs_lines'].obs_line_states

        return [robot_state_list, nei_state_list, obs_circular_list, obs_line_list]
    # # states
    # def total_states(self, env_train=True):
        
    #     robot_state_list = list(map(lambda r: np.squeeze( r.omni_state(env_train)), self.robot_list))
    #     nei_state_list = list(map(lambda r: np.squeeze( r.omni_obs_state(env_train)), self.robot_list))
    #     obs_circular_list = list(map(lambda o: np.squeeze( o.omni_obs_state(env_train) ), self.obs_cir_list))
    #     obs_line_list = self.obs_line_list
        
    #     return [robot_state_list, nei_state_list, obs_circular_list, obs_line_list]
        
    # def render(self, time=0.1, save=False, path=None, i = 0, **kwargs):
        
    #     self.world_plot.draw_robot_diff_list(**kwargs)
    #     self.world_plot.draw_obs_cir_list()
    #     self.world_plot.pause(time)

    #     if save == True:
    #         self.world_plot.save_gif_figure(path, i)

    #     self.world_plot.com_cla()

    
    # def seg_dis(self, segment, point):
        
    #     point = np.squeeze(point[0:2])
    #     sp = np.array(segment[0:2])
    #     ep = np.array(segment[2:4])

    #     l2 = (ep - sp) @ (ep - sp)

    #     if (l2 == 0.0):
    #         return np.linalg.norm(point - sp)

    #     t = max(0, min(1, ((point-sp) @ (ep-sp)) / l2 ))

    #     projection = sp + t * (ep-sp)

    #     distance = np.linalg.norm(point - projection) 

    #     return distance