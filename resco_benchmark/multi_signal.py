import os
import numpy as np
import traci
import sumolib
import gym
from traffic_signal import Signal
from utils.BB5B_sumo_methods import get_the_routes_info


class MultiSignal(gym.Env):
    def __init__(self, run_name, map_name, net, state_fn, reward_fn, route=None, gui=False, end_time=3600,
                 step_length=10, yellow_length=4, step_ratio=1, max_distance=200, lights=(), log_dir='/', libsumo=False,
                 warmup=0, gymma=False):
        self.libsumo = libsumo
        self.gymma = gymma  # gymma expects sequential list of states/rewards instead of dict
        print(map_name, net, state_fn.__name__, reward_fn.__name__)
        self.log_dir = log_dir
        self.net = net
        self.route = route
        self.gui = gui
        self.state_fn = state_fn
        self.reward_fn = reward_fn
        self.max_distance = max_distance
        self.warmup = warmup

        self.end_time = end_time
        self.step_length = step_length
        self.yellow_length = yellow_length
        self.step_ratio = step_ratio
        self.connection_name = run_name + '-' + map_name + '---' + state_fn.__name__ + '-' + reward_fn.__name__
        self.map_name = map_name
        self.run = None

        # Run some steps in the simulation with default light configurations to detect phases
        if self.route is not None:
            sumo_cmd = [sumolib.checkBinary('sumo'), '-n', net, '-r', self.route + '_1.rou.xml', '--no-warnings', 'True']
        else:
            sumo_cmd = [sumolib.checkBinary('sumo'), '-c', net, '--no-warnings', 'True']
        if self.libsumo:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label = self.connection_name)
            self.sumo = traci.getConnection(self.connection_name)
        self.signal_ids = self.sumo.trafficlight.getIDList()
        print('lights', len(self.signal_ids), self.signal_ids)
        valid_phases = dict()
        for i in range(0, 500):    # TODO grab info. directly from tllogic python interface
            for lightID in self.signal_ids:
                cur_phase = self.sumo.trafficlight.getRedYellowGreenState(lightID)
                if not lightID in valid_phases:
                    valid_phases[lightID] = []
                has_phase = False
                for phase in valid_phases[lightID]:
                    if phase == cur_phase:
                        has_phase = True
                if not has_phase:
                    valid_phases[lightID].append(cur_phase)
            self.step_sim()
        for ts in valid_phases:
            green_phases = []
            for phase in valid_phases[ts]:    # Convert to SUMO phase type
                if 'y' not in phase:
                    if phase.count('r') + phase.count('s') != len(phase):
                        green_phases.append(self.sumo.trafficlight.Phase(step_length, phase))
            valid_phases[ts] = green_phases

        self.phases = valid_phases

        self.signals = dict()

        self.all_ts_ids = lights if len(lights) > 0 else self.sumo.trafficlight.getIDList()
        self.ts_starter = len(self.all_ts_ids)
        self.signal_ids = []

        # Pull signal observation shapes
        self.obs_shape = dict()
        self.observation_space = list()
        self.action_space = list()
        for ts in self.all_ts_ids:
            self.signals[ts] = Signal(self.map_name, self.sumo, ts, self.yellow_length, self.phases[ts])
        for ts in self.all_ts_ids:
            self.signals[ts].signals = self.signals
            self.signals[ts].observe(self.step_length, self.max_distance)
        observations = self.state_fn(self.signals)
        self.ts_order = list()
        for ts in observations:
            if ts == 'top_mgr' or ts == 'bot_mgr': continue     # Not a traffic signal
            o_shape = observations[ts].shape
            self.obs_shape[ts] = o_shape
            o_shape = gym.spaces.Box(low=-np.inf, high=np.inf, shape=o_shape)
            self.ts_order.append(ts)
            self.observation_space.append(o_shape)
            self.action_space.append(gym.spaces.Discrete(len(self.phases[ts])))

        self.n_agents = self.ts_starter

        self.run = 0
        self.metrics = []
        self.wait_metric = dict()

        if not self.libsumo: traci.switch(self.connection_name)
        traci.close()
        self.connection_name = run_name + '-' + map_name + '-' + str(len(lights)) + '-' + state_fn.__name__ + '-' + reward_fn.__name__
        if not os.path.exists(log_dir+self.connection_name):
            os.makedirs(log_dir+self.connection_name)
        self.sumo_cmd = None
        print('Connection ID', self.connection_name)

        if self.map_name == "BB5B":
            self.INCOMING_ROADS_INFMain_Junc_DICT = {"N1": ["FMS2", "FMS1", "FMout"],
                                                     "N2": ["SHS2", "SHS1"],
                                                     "E": ["SKW21", "SKW2", "SKW1"],
                                                     "S1": ["INFN2", "INFN1", "INFout"],
                                                     "S2": ["TCN2", "TCN11", "TCN1", "TCEN"]
                                                     }
            self.INCOMING_ROADS_PBB_Junc_DICT = {"N": ["TMS2", "TMS1"],
                                                 "E": ["SHW2", "SHW1"],
                                                 "S1": ["SHN2", "SHN1", "HLout"],
                                                 "S2": ["FMN2", "FMN1"],
                                                 "W": ["HLE2", "HLE1"]
                                                 }
            self.INCOMING_ROADS_SIRIM_Junc_DICT = {"N1": ["TCS2", "TCS1", "TXout"],
                                                   "N2": ["INFS2", "INFS1", "INFS11", "SKWS"],
                                                   "E": ["SIRIMW21", "SIRIMW2", "SIRIMW1"],
                                                   "S": ["SIRIMN3", "SIRIMN2", "SIRIMN1"],
                                                   "W": ["TCE3", "TCE2", "TCE1"]
                                                   }

            self.current_total_waiting_time_on_incoming_lanes = 0
            self.old_total_waiting_time_vehicles_on_incoming_lanes = 0
            self.old_number_of_vehicles_that_passed_through_intersections = 0
            self.total_waiting_time_vehicles_on_incoming_lanes = 0
            self.old_total_wait = 0
            self.current_number_of_vehicles_that_passed_through_the_intersections_in_last_steps = 0
            self._waiting_times = {}
            self.waiting_time_vehicles_on_incoming_lanes = []
            self._reward_list_in_episode = []
            self._reward_mean_in_episode = []
            self.total_delays_of_all_vehicles_from_all_routes = []
            self.total_real_travel_times_all_vehicles_from_all_routes = []
            self.total_ideal_travel_times_all_vehicles_from_all_routes = []

            self.waiting_time_all_vehicles_in_simulation = []
            self.lanes_and_junctions_ids = []
            self.routes_info = {}
            self.vehicles_on_simulation = {}
            self.vehicles_on_incoming_lanes = dict()
            self.vehicles_on_outcoming_lanes = dict()

    def step_sim(self):
        # The monaco scenario expects .25s steps instead of 1s, account for that here.
        for _ in range(self.step_ratio):
            self.sumo.simulationStep()
            if self.map_name == "BB5B" and self.run is not None:
                self.update_waiting_time_all_vehicles_in_simulation()
                self.update_waiting_time_vehicles_on_incoming_lanes()
                self.update_waiting_time_vehicles_on_outcoming_lanes()

    def reset(self):
        if self.run != 0:
            if not self.libsumo: traci.switch(self.connection_name)
            traci.close()
            self.save_metrics()
        self.metrics = []

        self.run += 1

        # Start a new simulation
        self.sumo_cmd = []
        if self.gui:
            self.sumo_cmd.append(sumolib.checkBinary('sumo-gui'))
            self.sumo_cmd.append('--start')
        else:
            self.sumo_cmd.append(sumolib.checkBinary('sumo'))
        if self.route is not None:
            self.sumo_cmd += ['-n', self.net, '-r', self.route + '_'+str(self.run)+'.rou.xml']
        else:
            self.sumo_cmd += ['-c', self.net]
        self.sumo_cmd += ['--random', '--time-to-teleport', '-1', '--tripinfo-output',
                          os.path.join(self.log_dir, self.connection_name, 'tripinfo_' + str(self.run) + '.xml'),
                          '--tripinfo-output.write-unfinished',
                          '--no-step-log', 'True',
                          '--no-warnings', 'True']
        if self.libsumo:
            traci.start(self.sumo_cmd)
            self.sumo = traci
        else:
            traci.start(self.sumo_cmd, label=self.connection_name)
            self.sumo = traci.getConnection(self.connection_name)

        for _ in range(self.warmup):
            self.step_sim()

        # 'Start' only signals set for control, rest run fixed controllers
        if self.run % 30 == 0 and self.ts_starter < len(self.all_ts_ids): self.ts_starter += 1
        self.signal_ids = []
        for i in range(self.ts_starter):
            self.signal_ids.append(self.all_ts_ids[i])

        for ts in self.signal_ids:
            self.signals[ts] = Signal(self.map_name, self.sumo, ts, self.yellow_length, self.phases[ts])
            self.wait_metric[ts] = 0.0
        for ts in self.signal_ids:
            self.signals[ts].signals = self.signals
            self.signals[ts].observe(self.step_length, self.max_distance)

        if self.gymma:
            states = self.state_fn(self.signals)
            rets = list()
            for ts in self.ts_order:
                rets.append(states[ts])
            return rets

        if self.map_name == "BB5B":
            self.routes_info = get_the_routes_info()
            # get id all lanes and junctions
            self.lanes_and_junctions_ids = list(traci.lane.getIDList())

        return self.state_fn(self.signals)

    def step(self, act):
        if self.gymma:
            dict_act = dict()
            for i, ts in enumerate(self.ts_order):
                dict_act[ts] = act[i]
            act = dict_act

        # Send actions to their signals
        for signal in self.signals:
            self.signals[signal].prep_phase(act[signal])

        for step in range(self.yellow_length):
            self.step_sim()
        for signal in self.signal_ids:
            self.signals[signal].set_phase()
        for step in range(self.step_length - self.yellow_length):
            self.step_sim()
        for signal in self.signal_ids:
            self.signals[signal].observe(self.step_length, self.max_distance)

        # observe new state and reward
        observations = self.state_fn(self.signals)
        rewards = self.reward_fn(self.signals)

        self.calc_metrics(rewards)

        done = self.sumo.simulation.getTime() >= self.end_time

        info = {
            'step_time': self.sumo.simulation.getTime(),
             'observation': observations,
             'action': act,
             'reward': rewards,
             'current_number_of_vehicles': traci.vehicle.getIDCount(),
             'number_of_halting_vehicles_for_the_last_time_step_on_the_incoming_lanes': sum(self.signals[signal_id].get_total_queued() for signal_id in self.signal_ids),
             'waiting_time_for_the_last_time_step_on_the_incoming_lanes': sum([self.get_total_waiting_time_vehicles_on_incoming_lanes_per_lane(signal_id) for signal_id in self.signal_ids]),
             'number_of_all_halting_vehicles_for_the_last_time_step_in_simulation': self.get_total_queued_in_simulation(),
             'waiting_time_all_vehicles_for_the_last_time_step_in_simulation': round(self.get_total_waiting_time_in_simulation(), 5),
        }

        if self.gymma:
            obss, rww = list(), list()
            for ts in self.ts_order:
                obss.append(observations[ts])
                rww.append(rewards[ts])
            return obss, rww, [done], {'eps': self.run}
        return observations, rewards, done, {'eps': self.run}, info

    def calc_metrics(self, rewards):
        queue_lengths = dict()
        max_queues = dict()
        for signal_id in self.signals:
            signal = self.signals[signal_id]
            queue_length, max_queue = 0, 0
            for lane in signal.lanes:
                queue = signal.full_observation[lane]['queue']
                if queue > max_queue: max_queue = queue
                queue_length += queue
            queue_lengths[signal_id] = queue_length
            max_queues[signal_id] = max_queue
        self.metrics.append({
            'step': self.sumo.simulation.getTime(),
            'reward': rewards,
            'max_queues': max_queues,
            'queue_lengths': queue_lengths
        })

    def save_metrics(self):
        log = os.path.join(self.log_dir, self.connection_name+ os.sep + 'metrics_' + str(self.run) + '.csv')
        print('saving', log)
        with open(log, 'w+') as output_file:
            for line in self.metrics:
                csv_line = ''
                for metric in ['step', 'reward', 'max_queues', 'queue_lengths']:
                    csv_line = csv_line + str(line[metric]) + ', '
                output_file.write(csv_line + '\n')

    def render(self, mode='human'):
        pass

    def close(self):
        if not self.libsumo: traci.switch(self.connection_name)
        traci.close()
        self.save_metrics()

    # methods for BB5B scenario
    def update_waiting_time_all_vehicles_in_simulation(self):
        vehicle_list = traci.vehicle.getIDList()
        sim_step = traci.simulation.getTime()

        for veh_id in vehicle_list:
            veh_lane = traci.vehicle.getLaneID(veh_id)
            accumulated_waiting_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
            if veh_id not in self.vehicles_on_simulation:
                self.vehicles_on_simulation[veh_id] = {
                    "lanes": {veh_lane: accumulated_waiting_time},
                    "time": {"time_of_appearance": sim_step},
                    "routeID": traci.vehicle.getRouteID(veh_id),
                    "type": traci.vehicle.getTypeID(veh_id),
                    "max_speed": traci.vehicle.getMaxSpeed(veh_id),
                    "ideal_travel_time": self.routes_info[
                                             traci.vehicle.getRouteID(veh_id)
                                         ]["length"]
                                         / traci.vehicle.getMaxSpeed(veh_id),
                    "total_distance": traci.vehicle.getDistance(veh_id),
                    "last_calculate_delta_of_delays": False,
                }
            else:
                self.vehicles_on_simulation[veh_id]["lanes"][
                    veh_lane
                ] = accumulated_waiting_time - sum(
                    [
                        self.vehicles_on_simulation[veh_id]["lanes"][lane]
                        for lane in self.vehicles_on_simulation[veh_id]["lanes"].keys()
                        if lane != veh_lane
                    ]
                )

        # check if vehicle is not disappeared from road map
        for veh_id in list(self.vehicles_on_simulation.keys()):
            if (
                    veh_id not in vehicle_list
                    and "time_of_disappearance"
                    not in self.vehicles_on_simulation[veh_id]["time"].keys()
            ):
                self.vehicles_on_simulation[veh_id]["time"][
                    "time_of_disappearance"
                ] = sim_step
                self.vehicles_on_simulation[veh_id]["time"]["time_of_total_journey"] = (
                        self.vehicles_on_simulation[veh_id]["time"]["time_of_disappearance"]
                        - self.vehicles_on_simulation[veh_id]["time"]["time_of_appearance"]
                )

    def get_total_waiting_time_vehicles_on_incoming_lanes_per_lane(self, signal_id: str):
        wait_time_per_lane = []
        for lane in self.signals[signal_id].lanes:
            veh_list = [*traci.lane.getLastStepVehicleIDs(lane)]
            wait_time = 0.0
            for veh_id in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh_id)
                accumulated_waiting_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                if veh_id not in self.vehicles_on_incoming_lanes:
                    self.vehicles_on_incoming_lanes[veh_id] = {veh_lane: traci.vehicle.getWaitingTime(veh_id)}
                else:
                    self.vehicles_on_incoming_lanes[veh_id][veh_lane] = accumulated_waiting_time - sum([self.vehicles_on_simulation[veh_id]["lanes"][lane] for lane in self.vehicles_on_simulation[veh_id]["lanes"].keys() if lane != veh_lane])
                wait_time += traci.vehicle.getWaitingTime(veh_id)
            wait_time_per_lane.append(wait_time)
        return sum(wait_time_per_lane)

    def update_waiting_time_vehicles_on_incoming_lanes(self):
        for signal in self.signal_ids:
            for lane in self.signals[signal].lanes:
                veh_list = traci.lane.getLastStepVehicleIDs(lane)
                for veh_id in veh_list:
                    veh_lane = traci.vehicle.getLaneID(veh_id)
                    accumulated_waiting_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                    if veh_id not in self.vehicles_on_incoming_lanes:
                        self.vehicles_on_incoming_lanes[veh_id] = {veh_lane: traci.vehicle.getWaitingTime(veh_id)}
                    else:
                        self.vehicles_on_incoming_lanes[veh_id][veh_lane] = accumulated_waiting_time - sum([self.vehicles_on_simulation[veh_id]["lanes"][lane] for lane in self.vehicles_on_simulation[veh_id]["lanes"].keys() if lane != veh_lane])

    def update_waiting_time_vehicles_on_outcoming_lanes(self):
        sim_step = traci.simulation.getTime()
        for signal_id in self.signal_ids:
            for lane in self.signals[signal_id].outbound_lanes:
                veh_list = traci.lane.getLastStepVehicleIDs(lane)
                for veh_id in veh_list:
                    veh_lane = traci.vehicle.getLaneID(veh_id)
                    if veh_id not in self.vehicles_on_outcoming_lanes:
                        self.vehicles_on_outcoming_lanes[veh_id] = {
                            signal_id: {
                                veh_lane: sim_step
                            }
                        }
                    else:
                        if signal_id in self.vehicles_on_outcoming_lanes[veh_id]:
                            if veh_lane not in self.vehicles_on_outcoming_lanes[veh_id][signal_id]:
                                self.vehicles_on_outcoming_lanes[veh_id][signal_id][veh_lane] = sim_step
                        else:
                            self.vehicles_on_outcoming_lanes[veh_id][signal_id] = {
                                veh_lane: sim_step
                            }

    def get_total_queued_in_simulation(self):
        queue_list = []
        for id in self.lanes_and_junctions_ids:
            car_list = traci.lane.getLastStepVehicleIDs(id)
            for car_id in car_list:
                # new vehicles that appear in the simulation and have zero speed should not be considered in the queue
                if traci.vehicle.getSpeed(car_id) <= 0.1 and traci.vehicle.getWaitingTime(car_id) != 0:
                    queue_list.append(car_id)
        # return sum([traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in self.lanes_and_junctions_ids] if
        # traci.lane)
        return len(queue_list)

    def get_total_waiting_time_in_simulation(self):
        return sum([int(traci.lane.getWaitingTime(lane_id)) for lane_id in self.lanes_and_junctions_ids])
