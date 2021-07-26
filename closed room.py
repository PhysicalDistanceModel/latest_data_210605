#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import random
import copy
import pymunk as pm
from pymunk import Vec2d
import geopandas as gpd
from math import exp
from random import randint
from queue import Queue
import matplotlib.pyplot as plt
import math
import gym
import pygame
import pandas as pd
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
import numpy as np
import seaborn as sns
import time
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

for scence in [3.3, 3.2, 3.1]:
    boundry_shp_path = "env{}".format(scence) + "/boundry.shp"
    people_shp_path = "env{}".format(scence) + "/people.shp"
    wall_shp_path = "env{}".format(scence) + "/wall.shp"
    dest1_shp_path = "env{}".format(scence) + "/dest1.shp"
    dest2_shp_path = "env{}".format(scence) + "/dest2.shp"
    dest_shp_path = None


    class Config:
        def __init__(self):
            pd.set_option('display.unicode.ambiguous_as_wide', True)
            pd.set_option('display.unicode.east_asian_width', True)
            pd.set_option('display.max_columns', None)
            self.reset()
            boundry_bounds_list = self.GetBounds(self.boundry_shp_path)
            boundry_name_list = ["bminx", "bminy", "bmaxx", "bmaxy"]
            boundry_dict = {"b" + str(s): {boundry_name_list[a]: boundry_bounds_list[s][a] for a in range(4)} for s
                            in
                            range(len(boundry_bounds_list))}
            boundry_dict["bnum"] = len(boundry_bounds_list)
            people_bounds_list = self.GetBounds(self.people_shp_path)
            people_name_list = ["pminx", "pminy", "pmaxx", "pmaxy"]
            people_dict = {"p" + str(s) + "_" + str(list(people_bounds_list[s].keys())[0]):
                               {people_name_list[a]: list(people_bounds_list[s].values())[0][a] for a in range(4)}
                           for s in range(len(people_bounds_list))}
            people_dict["pnum"] = len(people_bounds_list)
            data = {**boundry_dict, **people_dict}
            self.people_boundry_dict = copy.deepcopy(people_dict)
            for i in range(data["bnum"]):
                self.bminx, self.bminy, self.bmaxx, self.bmaxy = data["b" + str(i)]["bminx"], data["b" + str(i)][
                    "bminy"], data["b" + str(i)]["bmaxx"], data["b" + str(i)]["bmaxy"]
            self.x, self.y = (self.bmaxx - self.bminx), (self.bmaxy - self.bminy)
            self.width = 500
            self.height = int((self.y / self.x) * self.width)
            self.wscale = self.width * (1 + ((self.bmaxy - self.bminy) / (self.bmaxx - self.bminx))) / (
                    (self.bmaxy - self.bminy) + (self.bmaxx - self.bminx))
            self.pscale = self.wscale
            self.color_dict = {}
            self.color_dict[list(people_bounds_list[0].keys())[0]] = SpaceDebugColor(102, 119, 238, 255)
            self.healthycolor = SpaceDebugColor(100, 100, 100, 255)
            self.exposurecolor = SpaceDebugColor(180, 85, 85, 255)
            self.illcolor = SpaceDebugColor(255, 106, 106, 255)

        def reset(self):
            self.boundry_shp_path = boundry_shp_path
            self.people_shp_path = people_shp_path
            self.COLLISION_AGENT = 1
            self.COLLISION_WALL = 2
            self.agent_num = 1
            self.SDM = True
            self.dt = 1.0 / 20.0
            self.breathcycle = 4.0
            self.physics_steps_per_frame = 1
            self.J = 10
            self.grid_size = (0.5, 0.5)
            self.people_target = 50
            self.interval = (5 / self.dt) * self.physics_steps_per_frame
            self.start_infection_num = 1
            self.ctl_radius = 1.5
            self.r = 4
            self.infection_radius = 2
            self.iso_percentage = 1
            self.infect_rate = 0
            self.pinf = 0.16
            self.gama = 4

        def GetBounds(self, shp_path):
            boundry = gpd.GeoDataFrame.from_file(shp_path)
            boundry = boundry.to_crs(epsg=32651)
            bounds_list = []
            for index, row in boundry.iterrows():
                bounds = row["geometry"].bounds
                try:
                    id_ = row["id"]
                    bounds_list.append({id_: bounds})
                except:
                    bounds_list.append(bounds)
            return bounds_list


    config = Config()


    class GenTool:
        def __init__(self, space, wall_shp_path=wall_shp_path,
                     dest_shp_path=dest_shp_path):
            self.space = space
            self.wall_shp_path = wall_shp_path
            self.dest_shp_path = dest_shp_path
            self.agents = []
            self.wall_list = []
            self.contact_time = {}
            self.density = 2000
            self.bminx, self.bminy, self.bmaxx, self.bmaxy = config.bminx, config.bminy, config.bmaxx, config.bmaxy
            self.people_boundry_dict = config.people_boundry_dict
            self.width, self.height = config.width, config.height
            self.wscale, self.pscale = config.wscale, config.pscale

        def GenAttr(self, people_boundry_id):
            if np.random.uniform(0, 1) >= 0.5:
                dSpeed = np.random.uniform(1.3, 1.56)
                mass = np.random.uniform(44, 83)
                radius = np.random.uniform(0.191, 0.243)
            else:
                dSpeed = np.random.uniform(1.2, 1.46)
                mass = np.random.uniform(38, 74)
                radius = np.random.uniform(0.173, 0.229)
            if random.uniform(0, 1) <= config.iso_percentage:
                pDistance = config.ctl_radius
            else:
                pDistance = 0
            for key in self.people_boundry_dict.keys():
                id_ = "p" + str(people_boundry_id) + "_"
                if id_ in key:
                    agent_id = str(key).split(id_)[-1]
                    minx, miny = self.norm(self.people_boundry_dict[key]["pminx"],
                                           self.people_boundry_dict[key]["pminy"])
                    maxx, maxy = self.norm(self.people_boundry_dict[key]["pmaxx"],
                                           self.people_boundry_dict[key]["pmaxy"])
                    x = random.uniform(minx, maxx)
                    y = random.uniform(miny, maxy)
            return Vec2d(x, y), dSpeed, mass, radius, pDistance, agent_id

        def GenAgent(self, pos, dSpeed, mass, radius, pDistance, agent_id):
            inertia = pm.moment_for_circle(mass, 0, radius, offset=(0, 0))
            body = pm.Body(mass, inertia)
            body.position = pos
            shape = pm.Circle(body, radius, (0, 0))
            shape.elasticity = 0
            shape.friction = 0
            shape.collision_type = config.COLLISION_AGENT
            shape.color = config.color_dict[agent_id]
            random_infect = random.uniform(0, 1)
            if random_infect <= config.infect_rate:
                infection = True
                shape.color = config.illcolor
            else:
                infection = False
            self.contact_time[shape] = {}
            self.space.add(body, shape)
            averageV = Vec2d.zero()
            energy = 0
            agent_power = Vec2d.zero()
            dest_count = 0
            agent_dict = {
                shape: [dSpeed, pDistance, infection, self.contact_time, averageV, energy, agent_power, dest_count,
                        agent_id]}
            return agent_dict

        def GenAgentCrowd(self, infect_num=config.start_infection_num):
            agents_dict = {}
            agents_found = 0
            for i in range(0, config.agent_num):
                found = False
                while not found:
                    people_boundry_id = np.random.randint(low=0, high=self.people_boundry_dict["pnum"])
                    pos, dSpeed, mass, radius, pDistance, agent_id = self.GenAttr(people_boundry_id)
                    if len(agents_dict) > 0:
                        countagents = 0
                        for shape in agents_dict.keys():
                            shape.color = config.color_dict[agent_id]
                            position = shape.body.position
                            dist = Vec2d((position[0] - pos[0]), (position[1] - pos[1])).length
                            if dist > shape.radius + radius:
                                countagents += 1
                        if countagents == i:
                            found = True
                            agents_found += 1
                            agent_dict = self.GenAgent(pos, dSpeed, mass, radius, pDistance, agent_id)
                            agents_dict.update(agent_dict)
                    else:
                        found = True
                        agents_found += 1
                        agent_dict = self.GenAgent(pos, dSpeed, mass, radius, pDistance, agent_id)
                        agents_dict.update(agent_dict)
            _ = 1
            while _ <= (infect_num):
                i = random.randint(0, len(agents_dict) - 1)
                shape = list(agents_dict.keys())[i]
                if agents_dict[shape][2]:
                    continue
                else:
                    agents_dict[shape][2] = True
                    shape.color = config.illcolor
                _ += 1
            return agents_dict

        def GetWallList(self):
            boundry = gpd.GeoDataFrame.from_file(self.wall_shp_path)
            boundry = boundry.to_crs(epsg=32651)
            for geometry in boundry.loc[:, "geometry"]:
                coords = geometry.coords
                for index in range(len(coords) - 1):
                    a = (self.norm(coords[index][0], coords[index][1]))
                    b = (self.norm(coords[index + 1][0], coords[index + 1][1]))
                    shape = self.GenWall(a, b)
                    self.wall_list.append(shape)
            return self.wall_list

        def GenWall(self, a, b):
            radius = 0.24
            moment = pm.moment_for_segment(0, a=a, b=b, radius=radius)
            body = pm.Body(0, moment, pm.Body.STATIC)
            shape = pm.Segment(body, a=a, b=b, radius=radius)
            shape.density = self.density
            shape.elasticity = 0
            shape.friction = 0.0
            shape.collision_type = config.COLLISION_WALL
            self.space.add(body, shape)
            return shape

        def GenDestList(self, dest_shp_path=None):
            if dest_shp_path == None:
                dest = gpd.GeoDataFrame.from_file(self.dest_shp_path)
            else:
                dest = gpd.GeoDataFrame.from_file(dest_shp_path)
            dest = dest.to_crs(epsg=32651)
            dest.loc[:, "Id"] = dest.index
            dest_list = []
            for index, row in dest.iterrows():
                coords = row["geometry"].coords
                id_ = row["id"]
                a = (self.norm(coords[0][0], coords[0][1]))
                b = (self.norm(coords[1][0], coords[1][1]))
                dest_list.append({str(index) + "_" + id_: (a, b)})
            return dest_list

        def norm(self, x, y):
            return (x - self.bminx), (y - self.bminy)

        def GetBounds(self, shp_path):
            boundry = gpd.GeoDataFrame.from_file(shp_path)
            boundry = boundry.to_crs(epsg=32651)
            for geometry in boundry.loc[:, "geometry"]:
                return geometry.bounds

        def reset(self):
            wall_list = self.GetWallList()
            agents_dict = self.GenAgentCrowd(infect_num=config.start_infection_num)
            dest_dict = self.GenDestList()
            return wall_list, agents_dict, dest_dict


    class Grid(object):
        def __init__(self, grid=(20, 20), compute=None):
            self.compute = compute
            self.steps = [(-1, 0), (-2, -1), (-1, -1), (-1, -2), (0, -1), (1, -2), (+1, -1), (2, -1)
                , (+1, 0), (2, 1), (+1, +1), (1, 2), (0, +1), (-1, 2), (-1, +1), (-2, 1)]
            self.grid = grid
            self.width = config.width
            self.height = config.height
            self.xgrid = self.getXGrid((0, 0, self.width, self.height))
            self.ygrid = self.getYGrid((0, 0, self.width, self.height))
            self.wallgrid = np.zeros([len(self.xgrid), len(self.ygrid)])
            self.rows = np.shape(self.wallgrid)[0]
            self.columns = np.shape(self.wallgrid)[1]
            self.costs = []
            self.directions = []
            self.step_direction = []
            self.addStepVector()
            self.dest_length = None
            self.dest_grid_loc = {}
            self.save_file = './grid{}/'.format(scence)
            if not os.path.exists(self.save_file):
                os.mkdir(self.save_file)

        def reset(self):
            self.dest = self.compute.dest_list
            self.dest_length = len(self.dest)
            self.destgrid = []
            for dict_ in self.dest:
                name = list(dict_.keys())[0]
                self.destgrid.append({name: np.zeros([len(self.xgrid), len(self.ygrid)])})
            for dir in os.listdir(self.save_file):
                if "maxdirection" in dir:
                    self.directions.append(
                        {str(dir).split("maxdirection")[-1].split(".npy")[0]: np.load(
                            os.path.join(self.save_file, dir))})
            self.destgrid = np.load(
                os.path.join(self.save_file, str(self.grid[0]) + 'maxdestgrid' + ".npy"), allow_pickle=True)
            for dict_ in self.destgrid:
                dest_pos = list(dict_.values())[0]
                dest_id = list(dict_.keys())[0]
                self.dest_grid_loc[dest_id] = []
                pre_target = np.asarray(np.where(dest_pos)).T
                for t in pre_target:
                    self.dest_grid_loc[dest_id].append((t[0], t[1]))

        def get_dest(self, x, y, count, agent_id):
            name = str(count) + "_" + str(agent_id)
            dest_grid_loc = self.dest_grid_loc[name][int(len(self.dest_grid_loc[name]) / 2)]
            dest = (Vec2d(dest_grid_loc[0] * config.grid_size[0],
                          dest_grid_loc[1] * config.grid_size[1]) - Vec2d(x, y)).normalized()
            return dest

        def getXGrid(self, coord):
            if coord[0] < coord[2]:
                xrange = []
                t = int(coord[0] / self.grid[0])
                xrange.append(t * self.grid[0])
                t += 1
                while t * self.grid[0] <= coord[2]:
                    xrange.append(t * self.grid[0])
                    t += 1
                xrange = np.asarray(xrange)
            else:
                xrange = []
                t = int(coord[2] / self.grid[0])
                xrange.append(t * self.grid[0])
                t += 1
                while t * self.grid[0] <= coord[0]:
                    xrange.append(t * self.grid[0])
                    t += 1
                xrange = np.asarray(xrange)
            return xrange

        def getYGrid(self, coord):
            if coord[1] < coord[3]:
                yrange = []
                t = int(coord[1] / self.grid[1])
                yrange.append(t * self.grid[1])
                t += 1
                while t * self.grid[1] <= coord[3]:
                    yrange.append(t * self.grid[1])
                    t += 1
                yrange = np.asarray(yrange)
            else:
                yrange = []
                t = int(coord[3] / self.grid[1])
                yrange.append(t * self.grid[1])
                t += 1
                while t * self.grid[1] <= coord[1]:
                    yrange.append(t * self.grid[1])
                    t += 1
                yrange = np.asarray(yrange)
            return yrange

        def getLoc(self, position, xrange, yrange):
            true_xval = max(xrange[position[0] >= xrange], default=0)
            true_yval = max(yrange[position[1] >= yrange], default=0)
            binlocx = np.where(xrange == true_xval)
            binlocy = np.where(yrange == true_yval)
            return (binlocx, binlocy)

        def addStepVector(self):
            for step in self.steps:
                self.step_direction.append(Vec2d(step[0], step[1]).normalized())
            self.step_direction.append(Vec2d(0, 0))


    class Engine(object):
        def __init__(self, shape, agents_dict, steps, dest, SDM):
            attr_list = agents_dict[shape]
            self.space = shape.space
            self.mass = shape.body.mass
            self.radius = shape.radius
            self.body = shape.body
            self.dSpeed_0 = attr_list[0]
            self.pDistance = attr_list[1]
            self.max_velocity = 1.5 * self.dSpeed_0
            self.min_velocity = 0.01 * self.dSpeed_0
            self.acclTime = .5
            self.pos = self.body.position
            self.direction = dest
            self.SDM = SDM
            self.steps = steps
            if self.steps > 0:
                self.averageV = attr_list[4].length / steps
            else:
                self.averageV = Vec2d.zero()
            self.dSpeed = self.dspeed()
            self.dVelocity = self.dSpeed * self.direction
            self.actualV = self.body.velocity
            self.bodyFactor = 120000
            self.slideFricFactor = 240000
            self.A = 2000
            self.B = 0.08
            self.AP = 5
            self.lambda_ = 0.2
            self.c = 2
            self.phi = 0.5

        def f1_ij(self, other):
            if self.cosine(other) >= math.cos(0.34 * math.pi):
                distance, _ = self.distance(other)
                func_fi = self.func_fi(self.pDistance + other.pDistance - 2 * distance)
                f1 = (func_fi * self.dVelocity - self.actualV) * (1 - func_fi * (self.pDistance / distance) ** 2)
                f1 = f1 * self.mass / self.acclTime
                return f1
            else:
                return self.f1()

        def func_fi(self, x):
            if x <= 0:
                return 1
            else:
                return 0

        def f1(self):
            deltaV = self.dVelocity - self.actualV
            f1 = (deltaV * self.mass) / self.acclTime
            return f1

        def peopleInteraction(self, other, dij):
            try:
                rij = self.radius + other.radius
                nij = (self.pos - other.pos).normalized()
                first = self.A * exp((rij - dij) / self.B) * nij
                return first
            except:
                return Vec2d.zero()

        def wallInteraction(self, wall_list):
            ri = self.radius
            diw, niw = self.distance2wall(self.pos, wall_list)
            if diw < 4:
                repulsive_force = self.A * exp((ri - diw) / self.B) * niw
                body_force = self.bodyFactor * self.g(ri - diw) * niw
                tiw = Vec2d(-niw[1], niw[0])
                slide = self.slideFricFactor * self.g(ri - diw) * (self.actualV.dot(tiw) * tiw)
                slide_force = Vec2d(slide[0], slide[1])
                f3 = repulsive_force + body_force - slide_force
                return f3
            else:
                return Vec2d.zero()

        def pDefense(self, other):
            distance, nij = self.distance(other)
            cos_theta = -nij.dot(self.actualV.normalized())
            fd = nij * self.AP * 2 * (self.pDistance - distance / self.pDistance + other.pDistance) * (
                    self.lambda_ + (1 - self.lambda_) * (1 + cos_theta) / 2)
            return fd

        def distance(self, other):
            rij = self.radius + other.radius
            dij = (self.pos - other.pos).length
            nij = (self.pos - other.pos).normalized()
            return dij - rij, nij

        def cosine(self, other):
            a = self.direction.normalized()
            b = (other.pos - self.pos).normalized()
            c = Vec2d.dot(a, b)
            return c

        def dspeed(self):
            if self.steps > 0 and self.SDM:
                na = 1 - (self.averageV / self.dSpeed_0)
                return (1 - na) * self.dSpeed_0 + na * self.max_velocity
            else:
                return self.dSpeed_0

        def g(self, x):
            return np.maximum(x, 0)

        def distance2wall(self, point, wall):
            p0 = Vec2d(wall.a[0], wall.a[1])
            p1 = Vec2d(wall.b[0], wall.b[1])
            d = p1 - p0
            ymp0 = point - p0
            t = Vec2d.dot(d, ymp0) / Vec2d.dot(d, d)
            if t <= 0.0:
                dist = ymp0.length
                cross = p0 + t * d
            elif t >= 1.0:
                ymp1 = point - p1
                dist = ymp1.length
                cross = p0 + t * d
            else:
                cross = p0 + t * d
                dist = (cross - point).length
            npw = Vec2d.normalized(point - cross)
            return dist, npw


    class Compute(object):
        def __init__(self, SDM):
            self.infection = None
            self.steps = None
            self.SDM = SDM
            self.agents_dict = {}
            self.dest_index = {}
            self.trajectory = {}
            self.trajectory_copy = {}
            self.dest_list = None
            self.wall_list = None
            self.grid = None

        def genagent(self, people_boundry_id, gen=None):
            pos, dSpeed, mass, radius, pDistance, agent_id = gen.GenAttr(people_boundry_id)
            agent_dict = gen.GenAgent(pos, dSpeed, mass, radius, pDistance, agent_id)
            self.agents_dict.update(agent_dict)
            self.dest_index[list(agent_dict.keys())[0]] = 0
            self.trajectory[list(agent_dict.keys())[0]] = []

        def reset(self, infect_num=config.start_infection_num, gen=None, crowd=False):
            self.steps = 0
            self.agents_dict = {}
            self.dest_index = {}
            self.trajectory = {}
            self.trajectory_copy = {}
            self.exposure_num = 0
            self.recorder_ok = False
            self.ill_exist = None
            self.people_contact_num = 0
            if os.path.exists(dest2_shp_path):
                dest1 = gen.GenDestList(dest1_shp_path)
                dest2 = gen.GenDestList(dest2_shp_path)
                self.dest_list = dest1 + dest2
            else:
                self.dest_list = gen.GenDestList(dest1_shp_path)
            self.wall_list = gen.GetWallList()
            if crowd:
                agents_dict = gen.GenAgentCrowd(infect_num)
                self.agents_dict.update(agents_dict)
                for shape in agents_dict.keys():
                    self.dest_index[shape] = 0
                    self.trajectory[shape] = []

        def update_infection(self, shape):
            self.ill_exist = False
            if self.agents_dict.get(shape)[2]:
                if not shape.color == config.illcolor:
                    shape.color = config.exposurecolor
                else:
                    self.ill_exist = True

        def dest_count_update(self, shapei):
            pos = shapei.body.position
            loc = np.asarray(self.grid.getLoc((pos[0], pos[1]), self.grid.xgrid, self.grid.ygrid)).T
            a = self.agents_dict[shapei][7]
            agent_id = self.agents_dict[shapei][8]
            name = str(a) + "_" + agent_id
            for dict_ in self.grid.destgrid:
                if list(dict_.keys())[0] == name:
                    if dict_[name][(loc[0][0][0], loc[0][0][1])] == 1.0:
                        a = self.agents_dict[shapei][7] + 1
                    if not os.path.exists(dest2_shp_path):
                        if a == self.grid.dest_length:
                            return -1
                    else:
                        if a == self.grid.dest_length / 2:
                            return -1
            self.agents_dict[shapei][7] = a
            return 0

        def infection_judegment(self, shapei, shapej, dis, current_time):
            if dis < config.ctl_radius:
                self.people_contact_num += 1
            if shapej.color == config.illcolor and self.agents_dict[shapei][
                2] == False:
                square_r = pow(config.infection_radius, 2)
                d = math.exp(-1 * pow(dis, 2) / square_r) / (square_r * math.pi)
                contact_dict = self.agents_dict[shapei][3][shapei]
                if shapej in contact_dict.keys():
                    contact_dict[shapej] += config.pinf * d * config.dt
                    if current_time % config.breathcycle >= (
                            config.breathcycle - config.dt) and current_time > 0:
                        p = 1 - np.exp(-1 * config.gama * contact_dict[shapej])
                        if np.random.uniform(0, 1) < p:
                            self.agents_dict[shapei][2] = True
                            self.exposure_num += 1
                        else:
                            contact_dict[shapej] = 0
                else:
                    contact_dict[shapej] = d * config.dt

        def step_calculation(self, current_time):
            current_time = round(current_time, 4)
            remove_list = []
            shape_position_list = []
            shape_list = []
            for shapei in self.agents_dict.keys():
                self.update_infection(shapei)
                shape_position_list.append(shapei.body.position)
                shape_list.append(shapei)
                if self.dest_count_update(shapei) == -1:
                    remove_list.append(shapei)
                self.trajectory[shapei].append(
                    (shapei.body.position.x * config.pscale, shapei.body.position.y * config.pscale))
            shape_position_list = np.array(shape_position_list)
            kdtree = KDTree(shape_position_list, leaf_size=12)
            nbrs = NearestNeighbors(n_neighbors=2, radius=4, leaf_size=20, algorithm='ball_tree').fit(
                shape_position_list)
            min_distances, indices = nbrs.kneighbors(shape_position_list)
            min_distance_index_list = indices[:, 1]
            for i_index in range(len(shape_list)):
                shapei = shape_list[i_index]
                dest = self.grid.get_dest(shapei.body.position[0], shapei.body.position[1],
                                          self.agents_dict[shapei][7],
                                          self.agents_dict[shapei][8])
                if dest == None:
                    dest = Vec2d.zero()
                agenti = Engine(shapei, self.agents_dict, self.steps, dest, self.SDM)
                try:
                    self.agents_dict[shapei][4] += agenti.actualV
                except:
                    pass
                wall_interaction = Vec2d.zero()
                people_interaction = Vec2d.zero()
                pDefense = Vec2d.zero()
                NearAgentsDisList = []
                if len(self.agents_dict) > 1:
                    agent_position = shape_position_list[i_index]
                    agent_position = np.expand_dims(agent_position, axis=0)
                    shapej_indexs = kdtree.query_radius(agent_position, r=4)
                    agentj_min_index = min_distance_index_list[i_index]
                    agentj_min = Engine(shape_list[agentj_min_index], self.agents_dict, self.steps,
                                        Vec2d.zero(), self.SDM)
                    dji_min, nji = agentj_min.distance(agenti)
                    if agenti.pDistance > 0:
                        pDefense += agenti.pDefense(agentj_min)
                        f1 = agenti.f1_ij(agentj_min)
                    else:
                        f1 = agenti.f1()
                    NearAgentsDisList.append(dji_min)
                    self.infection_judegment(shapei, shape_list[agentj_min_index], dji_min, current_time)
                    for shapej_index in shapej_indexs[0]:
                        if shapej_index == agentj_min_index or shapej_index == i_index:
                            continue
                        shapej = shape_list[shapej_index]
                        agentj = Engine(shapej, self.agents_dict, self.steps, Vec2d.zero(), self.SDM)
                        dis, nij = agenti.distance(agentj)
                        NearAgentsDisList.append(dis)
                        people_interaction += agenti.peopleInteraction(agentj, dis)
                        self.infection_judegment(shapei, shapej, dis, current_time)
                else:
                    f1 = agenti.f1()
                    agent_recorder.add_distance(0)
                for wall_shape in self.wall_list:
                    wall_interaction += agenti.wallInteraction(wall_shape)
                try:
                    if people_interaction.length > 100:
                        scale = 100 / people_interaction.length
                        people_interaction = people_interaction * scale
                except:
                    people_interaction = Vec2d.zero()
                if not self.SDM:
                    sumForce = agenti.f1() + people_interaction + wall_interaction
                else:
                    sumForce = f1 + people_interaction + wall_interaction + pDefense
                self.agents_dict[shapei][6] = sumForce
            return remove_list


    class Render_:
        def __init__(self, gen, compute):
            pygame.display.init()
            pygame.font.init()
            pygame.joystick.init()
            self.gen = gen
            self.height = config.height
            self.width = config.width
            self.compute = compute
            self.wall_contact_num = 0
            self.people_contact_num = 0
            self.font = pygame.font.SysFont("SimHei", 10)
            self.screen = pygame.display.set_mode((config.width, config.height), flags=pygame.HIDDEN)
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.wscale, self.pscale = config.wscale, config.pscale
            self.trajectory_ok = False

        def reset(self):
            self.trajectory_ok = False
            self.screen = pygame.display.set_mode((config.width, config.height))
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        def render_(self, dt, render_set=True):
            self.screen.fill(pygame.Color("white"))
            for wall in self.compute.wall_list:
                self.draw_options.draw_fat_segment(a=(wall.a.x * self.wscale, wall.a.y * self.wscale),
                                                   b=(wall.b.x * self.wscale, wall.b.y * self.wscale),
                                                   radius=1,
                                                   outline_color=SpaceDebugColor(0, 0, 0, 255),
                                                   fill_color=SpaceDebugColor(0, 0, 0, 255))
            if self.trajectory_ok:
                histry_trajectory = self.compute.trajectory_copy
                if histry_trajectory:
                    histry_trajectorys = list(histry_trajectory.values())
                    if len(histry_trajectorys) > 50:
                        for trajectory in histry_trajectorys[-50:]:
                            pygame.draw.lines(self.screen, SpaceDebugColor(220, 220, 220, 50), False, trajectory)
                    else:
                        for trajectory in histry_trajectorys:
                            pygame.draw.lines(self.screen, SpaceDebugColor(220, 220, 220, 50), False, trajectory)
            for shape in self.compute.agents_dict.keys():
                trajectory = self.compute.trajectory[shape]
                if len(trajectory) >= 2 and self.trajectory_ok:
                    pygame.draw.lines(self.screen, SpaceDebugColor(220, 220, 220, 50), False, trajectory)
                radius = shape.radius
                position = shape.body.position
                p = Vec2d(position.x * self.pscale, position.y * self.pscale)
                if self.trajectory_ok:
                    self.draw_options.draw_circle(pos=p, angle=0, radius=radius * self.pscale,
                                                  outline_color=SpaceDebugColor(255, 255, 255, 255),
                                                  fill_color=shape.color)
                    pygame.draw.circle(self.screen, shape.color, p, radius * self.pscale, 1)
                else:
                    pygame.draw.circle(self.screen, shape.color, p, radius * self.pscale, 1)
            self.clock.tick(1 / dt)
            if render_set:
                pygame.display.flip()
            self.event()

        def event(self):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.K_f:
                    pygame.quit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    pygame.image.save(self.screen, "quickshot.png")


    class Environment(gym.Env):
        def __init__(self, SDM=config.SDM):
            self.outflowJ = int(config.J / config.physics_steps_per_frame) * 100
            self.total_time = None
            self.previous_time = None
            self.curr_score = None
            self.grid_size = config.grid_size
            self.grid = Grid(self.grid_size, None)
            self.action_list = None
            self.space = None
            self.state = None
            self.total_reward = None
            self.people_contact_num = None
            self.wall_contact_num = None
            self.compute = Compute(SDM)
            self.width, self.height = config.width, config.height
            self.grid_num = self.grid.wallgrid.ravel().shape[0]

        def render_(self, dt=config.dt * config.physics_steps_per_frame):
            if self.render_ok == 0:
                self.render_engine.reset()
                self.render_ok = 1
            self.render_engine.render_(dt)
            self.people_contact_num = self.render_engine.people_contact_num
            self.wall_contact_num = self.render_engine.wall_contact_num

        def remove_list(self, remove_list):
            for shape in remove_list:
                self.space.remove(shape.body, shape)
                self.compute.trajectory_copy[shape] = self.compute.trajectory[shape]
                del self.compute.agents_dict[shape]

        def close(self):
            return pygame.quit(), pygame.display.quit()

        def reset(self):
            self.total_time = 0
            self.previous_time = 0
            self.total_reward = 0
            self.curr_score = 0
            self.curr_exposure = 0
            self.noops = 0
            self.shusan_num = {}
            self.all_num = {}
            self.eva_num = {}
            self.render_ok = 0
            self.people_contact_num = 0
            self.wall_contact_num = 0
            self.done_ok = 0
            del self.space
            self.space = pm.Space()
            self.space.gravity = 0, 0
            self.space.collision_bias = 0.1
            self.space.collision_persistence = 2
            self.space.collision_slop = 0.1
            self.space.iterations = 5
            self.infect_num = config.start_infection_num
            self.gen = GenTool(self.space)
            self.compute.reset(self.infect_num, self.gen, crowd=True)
            self.grid = Grid(self.grid_size, self.compute)
            self.grid.reset()
            self.compute.grid = self.grid
            self.render_engine = Render_(self.gen, self.compute)
            self.info = {"time": 0}
            return self.state

        def step(self, action=None, dt=config.dt):
            for _ in range(config.physics_steps_per_frame):
                self.space.step(dt)
                self.total_time += self.space.current_time_step
            if action != self.gen.people_boundry_dict["pnum"]:
                self.compute.genagent(people_boundry_id=action, gen=self.gen)
            for shape in self.compute.agents_dict.keys():
                impulse = self.compute.agents_dict[shape][6]
                shape.body.apply_force_at_world_point(impulse, shape.body.position)
            remove_list = self.compute.step_calculation(self.total_time)
            self.remove_list(remove_list)
            observation = None
            reward = None
            done = bool(len(self.compute.agents_dict) <= 1 or self.total_time > 60)
            self.render_engine.trajectory_ok = True
            self.info["time"] = self.total_time
            self.compute.steps += 1
            return observation, reward, done, self.info

    if __name__ == "__main__":
        iso_pencentage_list = np.linspace(0, 1, num=6, endpoint=True)
        for iso_pencentage in iso_pencentage_list:
            config.reset()
            config.iso_percentage = iso_pencentage
            config.agent_num = 120
            env = Environment()
            env.reset()
            done = False
            while not done:
                observation, reward, done, info = env.step(action=config.people_boundry_dict['pnum'])
                env.render_()

