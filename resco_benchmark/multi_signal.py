import codecs
import os
from pathlib import Path
from statistics import mean, stdev

import gym
import numpy as np
import sumolib
import traci
import traci.constants as tc
import yaml
import logging

import rewards
from traffic_signal import Signal
from utils.BB5B_sumo_methods import get_the_routes_info
from utils.mytl.generators.routes import RoutesGenerator

logger = logging.getLogger('resco_benchmark')


class MultiSignal(gym.Env):
    def __init__(self, run_name, map_name, net, state_fn, reward_type,
                 validation_day_directory_name: str,
                 validation_period_file_name: str, route=None,
                 gui=False, start_time=0, end_time=3600, step_length=10,
                 yellow_length=3, step_ratio=1, max_distance=200,
                 lights=(), log_dir='/', libsumo=False, warmup=0,
                 gymma=False, mode="training", seed=42):
        self.libsumo = libsumo
        self.gymma = gymma  # gymma expects sequential list of states/rewards instead of dict

        self.reward_fn = getattr(rewards, reward_type)
        print(map_name, net, state_fn.__name__, reward_type)
        self.run_name = run_name
        self.log_dir = log_dir
        self.net = net
        self.route = route
        self.gui = gui
        self.state_fn = state_fn
        self.max_distance = max_distance
        self.warmup = warmup

        self.start_time = start_time
        self.end_time = end_time
        self.step_length = step_length
        self.yellow_length = yellow_length
        self.red_length = 2
        self.step_ratio = step_ratio
        self.connection_name = run_name + '-' + map_name + '---' + state_fn.__name__ + '-' + reward_type
        self.map_name = map_name
        self.run = None
        self.mode = mode
        self.seed = seed

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

        self.phases = {
            lightID: [
                p
                for p in self.sumo.trafficlight.getAllProgramLogics(lightID)[0].getPhases()
                if "y" not in p.state and "g" in p.state.lower() and p.duration > 3.0
            ]
            for lightID in self.signal_ids
        }

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
            # if ts == 'top_mgr' or ts == 'bot_mgr': continue     # Not a traffic signal
            o_shape = observations[ts].shape
            self.obs_shape[ts] = o_shape
            o_shape = gym.spaces.Box(low=-np.inf, high=np.inf, shape=o_shape)
            self.ts_order.append(ts)
            self.observation_space.append(o_shape)
            # self.action_space.append(gym.spaces.Discrete(len(self.phases[ts])))
            self.action_space.append(gym.spaces.Discrete(2))

        self.n_agents = self.ts_starter

        self.run = 0
        self.metrics = []
        self.wait_metric = dict()

        if not self.libsumo: traci.switch(self.connection_name)
        traci.close()
        self.connection_name = run_name + '-' + map_name + '-' + str(len(lights)) + '-' + state_fn.__name__ + '-' + reward_type
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
            self.validation_day_directory_name = validation_day_directory_name
            self.validation_period_file_name = self.map_name + "_" + validation_period_file_name + ".rou.xml"

    def step_sim(self):
        # The monaco scenario expects .25s steps instead of 1s, account for that here.
        for _ in range(self.step_ratio):
            if self.map_name == "BB5B" and self.sim_step == 25201:
                traci.trafficlight.setPhase("INFMain_Junc", 5)
            self.sumo.simulationStep()
            self.update_sim_step() #TODO: better name
            self.reset_traci_subscriptions()
            
            if self.map_name == "BB5B" and self.run is not None:
                for ts in self.signal_ids:
                    self.signals[ts].update()
                self.update_waiting_time_all_vehicles_in_simulation()
                self.check_if_vehicle_has_not_disappeared_from_environment()
                self.update_waiting_time_vehicles_on_incoming_lanes()
                self.check_if_vehicle_pass_the_junctions()

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
            if self.map_name == "BB5B":
                if self.mode == "training":
                    # generate train file with routes
                    self.route = self.generate_training_file_with_routes()
                else: # if mode = validation
                    self.route = Path("environments", self.map_name, self.validation_day_directory_name, self.validation_period_file_name)

                self.sumo_cmd += [
                    '--begin', str(self.start_time),
                    '--end', str(self.end_time)
                ]

                self.sumo_cmd += ['-c', self.net, '-r', str(self.route)]
            else:
                self.sumo_cmd += ['-c', self.net]

        self.sumo_cmd += ['--time-to-teleport', '-1', '--tripinfo-output',
                          os.path.join(self.log_dir, self.connection_name, 'tripinfo_' + str(self.run) + '.xml'),
                          '--tripinfo-output.write-unfinished',
                          '--no-step-log', 'True',
                          '--no-warnings', 'True',
                          '--waiting-time-memory=4000',
                          '--seed', f"{self.seed}"] # '--random', 
        if self.libsumo:
            traci.start(self.sumo_cmd)
            self.sumo = traci
        else:
            traci.start(self.sumo_cmd, label=self.connection_name)
            self.sumo = traci.getConnection(self.connection_name)

        self.setup_traci_subscriptions()
        self.update_sim_step()

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
            self.vehicles_on_simulation = {}
            self.vehicles_on_incoming_lanes = dict()
            self.vehicles_on_outcoming_lanes = dict()

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

        for step in range(self.red_length + self.yellow_length):
            self.step_sim()
        for signal in self.signal_ids:
            self.signals[signal].set_phase()
        for step in range(self.step_length - self.yellow_length - self.red_length):
            self.step_sim()
        for signal in self.signal_ids:
            self.signals[signal].observe(self.step_length, self.max_distance)

        # observe new state and reward
        observations = self.state_fn(self.signals)
        rewards = self.reward_fn(self.signals)

        self.calc_metrics(rewards)

        # Increase simulation time by 4 (240 s) additional minutes to the vehicles, which appeared at the end of the
        # simulation, had a chance to complete their routes
        done = self.sim_step >= (self.end_time + 240)
        if done and self.map_name == "BB5B":
            self.calculate_travel_time_and_delays()
            count_of_vehicles_completing_journey = self.get_number_of_vehicles_completing_journey()
            total_time_of_journey = self.get_total_time_of_journey()

            info = {
                'step_time': self.sim_step,
                'observation': observations,
                'action': act,
                'reward': rewards,
                'eps': self.run,
                'count_of_all_vehicles_in_simulation': len(self.vehicles_on_simulation.keys()),
                'number_of_halting_vehicles_for_the_last_time_step_on_the_incoming_lanes': sum(self.signals[signal_id].get_total_queued() for signal_id in self.signal_ids),
                # 'waiting_time_for_the_last_time_step_on_the_incoming_lanes': sum([self.get_total_waiting_time_vehicles_on_incoming_lanes_per_lane(signal_id) for signal_id in self.signal_ids]),
                'number_of_all_halting_vehicles_for_the_last_time_step_in_simulation': self.get_total_queued_in_simulation(),
                'waiting_time_all_vehicles_for_the_last_time_step_in_simulation': round(self.get_total_waiting_time_in_simulation(), 5),
                'total_waiting_time_on_the_incoming_lanes_in_episode': sum(self.waiting_time_vehicles_on_incoming_lanes),
                'total_waiting_time_on_the_incoming_lanes_in_episode2': sum([self.get_total_waiting_time_vehicles_on_incoming_lanes_per_lane(signal_id) for signal_id in self.signal_ids]),
                'count_of_vehicles_completing_journey': count_of_vehicles_completing_journey,
                'total_time_of_journey': total_time_of_journey,
                'average_time_of_journey': round(total_time_of_journey/count_of_vehicles_completing_journey if count_of_vehicles_completing_journey != 0 else 0, 5),
                'total_sum_delays_of_all_vehicles_from_all_routes': round(sum(self.total_delays_of_all_vehicles_from_all_routes), 5),
                'total_average_delays_of_all_vehicles_from_all_routes': round(mean(self.total_delays_of_all_vehicles_from_all_routes) if len(self.total_delays_of_all_vehicles_from_all_routes) != 0 else 0, 5),
                'total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey': round(mean(self.total_delays_of_all_vehicles_from_all_routes + self.calculate_current_delays_of_all_vehicles_in_simulation()) if len(self.total_delays_of_all_vehicles_from_all_routes + self.calculate_current_delays_of_all_vehicles_in_simulation()) != 0 else 0, 5),
                'total_average_delays_real_times_by_ideal_times': round(sum(self.total_real_travel_times_all_vehicles_from_all_routes) / sum(self.total_ideal_travel_times_all_vehicles_from_all_routes), 5) if len(self.total_ideal_travel_times_all_vehicles_from_all_routes) != 0 else 0,
                'total_average_delays_with_weights': round(self.get_total_average_delays_with_weights(), 5),
                "routes": self.routes_info
            }
            self.route = None
        elif not done and self.map_name == "BB5B":
            current_waiting_time_vehicles_on_incoming_lanes = sum([self.get_total_waiting_time_vehicles_on_incoming_lanes_per_lane(signal_id) for signal_id in self.signal_ids])
            self.waiting_time_vehicles_on_incoming_lanes.append(current_waiting_time_vehicles_on_incoming_lanes)
            info = {
                'step_time': self.sim_step,
                'observation': observations,
                'action': act,
                'reward': rewards,
                'eps': self.run,
                'current_number_of_vehicles': traci.vehicle.getIDCount(),
                'number_of_halting_vehicles_for_the_last_time_step_on_the_incoming_lanes': sum(self.signals[signal_id].get_total_queued() for signal_id in self.signal_ids),
                # 'waiting_time_for_the_last_time_step_on_the_incoming_lanes': current_waiting_time_vehicles_on_incoming_lanes,
                'number_of_all_halting_vehicles_for_the_last_time_step_in_simulation': self.get_total_queued_in_simulation(),
                'waiting_time_all_vehicles_for_the_last_time_step_in_simulation': round(self.get_total_waiting_time_in_simulation(), 5),
                'current_average_delays_of_all_vehicles_in_simulation': round(mean(self.calculate_current_delays_of_all_vehicles_in_simulation()) if len(self.calculate_current_delays_of_all_vehicles_in_simulation()) != 0 else 0, 5),
                'calculate_average_delta_of_delays_after_action': round(self.calculate_delta_of_delays(), 5),
                'number_of_vehicles_that_passed_through_the_intersections_in_last_steps': self.calculate_the_number_of_vehicles_that_passed_through_the_intersections_in_last_steps(),
            }
        else:
            info = {
                'observation': observations,
                'action': act,
                'reward': rewards,
                'eps': self.run,
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
            'step': self.sim_step,
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

    def update_sim_step(self):
        self.sim_step = traci.simulation.getTime()

    # @property
    # def sim_step(self):
    #     """
    #     Return current simulation second on SUMO
    #     """
    #     return traci.simulation.getTime()

    # methods for BB5B scenario
    def update_waiting_time_all_vehicles_in_simulation(self):
        """
        This method is called in every sim step for BB5B scenario and updates
        the waiting times of all vehicles present in the environment.
        """

        scResults = self.get_traci_subscription(self._all_vehicles_subscription)

        if scResults is None:
            logger.warning("update_waiting_time_all_vehicles_in_simulation got no results")
        
        for veh_id, d in scResults.items():
            veh_lane = d[tc.VAR_LANE_ID]
            accumulated_waiting_time = d[tc.VAR_ACCUMULATED_WAITING_TIME]
            if veh_id not in self.vehicles_on_simulation:
                route_id = d[tc.VAR_ROUTE_ID]
                max_speed = d[tc.VAR_MAXSPEED]
                self.vehicles_on_simulation[veh_id] = {
                    "lanes": {veh_lane: accumulated_waiting_time},
                    "time": {"time_of_appearance": self.sim_step},
                    "routeID": route_id,
                    "type": d[tc.VAR_TYPE],
                    "max_speed": max_speed,
                    "ideal_travel_time": self.routes_info[route_id]["length"] / max_speed,
                    "total_distance": d[tc.VAR_DISTANCE],
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

    def setup_traci_subscriptions(self):
        junctionID = traci.junction.getIDList()[0]
        # subscribe around that junction with a sufficiently large
        # radius to retrieve the speeds of all vehicles in every step
        traci.junction.subscribeContext(
            junctionID, tc.CMD_GET_VEHICLE_VARIABLE, 1000000, #TODO: how big should this number be?
            [tc.VAR_SPEED, tc.VAR_MAXSPEED, tc.VAR_ROUTE_ID, tc.VAR_LANE_ID, tc.VAR_TYPE, tc.VAR_ACCUMULATED_WAITING_TIME, tc.VAR_DISTANCE, tc.VAR_WAITING_TIME]
        )
        self._all_vehicles_subscription = junctionID
        self._traci_subscriptions = {junctionID: None}

    def reset_traci_subscriptions(self):
        for subscription in self._traci_subscriptions:
            self._traci_subscriptions[subscription] = None

    def get_traci_subscription(self, subscription_id):
        res = self._traci_subscriptions[subscription_id]
        if res is None:
            res = traci.junction.getContextSubscriptionResults(subscription_id)
            self._traci_subscriptions[subscription_id] = res
        return res

    def calculate_the_number_of_vehicles_that_passed_through_the_intersections_in_last_steps(self):
        veh_passed_list = []
        for veh_id in self.vehicles_on_outcoming_lanes.keys():
            for signal_id in self.vehicles_on_outcoming_lanes[veh_id].keys():
                if list(self.vehicles_on_outcoming_lanes[veh_id][signal_id].values())[0] in np.arange((self.sim_step - self.step_length), self.sim_step):
                    # insert into the table the indexes of vehicles that appeared on exit lanes
                    # with intersections within the delta_time interval, e.g. 5 seconds
                    veh_passed_list.append(veh_id)
        return len(veh_passed_list)

    def check_if_vehicle_has_not_disappeared_from_environment(self):
        """
        This method is called in every sim step for BB5B scenario and checks
         if the vehicle has not disappeared from the road map.
        """
        scResults = self.get_traci_subscription(self._all_vehicles_subscription)
        vehicle_list = scResults.keys()

        for veh_id in self.vehicles_on_simulation.keys():
            if (
                    veh_id not in vehicle_list
                    and "time_of_disappearance"
                    not in self.vehicles_on_simulation[veh_id]["time"].keys()
            ):
                self.vehicles_on_simulation[veh_id]["time"][
                    "time_of_disappearance"
                ] = self.sim_step
                self.vehicles_on_simulation[veh_id]["time"]["time_of_total_journey"] = (
                        self.vehicles_on_simulation[veh_id]["time"]["time_of_disappearance"]
                        - self.vehicles_on_simulation[veh_id]["time"]["time_of_appearance"]
                )

    def get_total_waiting_time_vehicles_on_incoming_lanes_per_lane(self, signal_id: str):
        wait_time_per_lane = []
        for veh_lane in self.signals[signal_id].lanes:
            veh_list = [*traci.lane.getLastStepVehicleIDs(veh_lane)]
            wait_time = 0.0
            for veh_id in veh_list:
                accumulated_waiting_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                if veh_id not in self.vehicles_on_incoming_lanes:
                    self.vehicles_on_incoming_lanes[veh_id] = {veh_lane: traci.vehicle.getWaitingTime(veh_id)}
                else:
                    self.vehicles_on_incoming_lanes[veh_id][veh_lane] = accumulated_waiting_time - sum([self.vehicles_on_simulation[veh_id]["lanes"][lane] for lane in self.vehicles_on_simulation[veh_id]["lanes"].keys() if lane != veh_lane])
                wait_time += traci.vehicle.getWaitingTime(veh_id)
            wait_time_per_lane.append(wait_time)
        return sum(wait_time_per_lane)

    def update_waiting_time_vehicles_on_incoming_lanes(self):
        scResults = self.get_traci_subscription(self._all_vehicles_subscription)

        for signal in self.signal_ids: 
            for in_lane in self.signals[signal].lanes:
                # get the ids of the vehicles for the last time step on the given lane
                veh_list = list(traci.lane.getLastStepVehicleIDs(in_lane))#TODO: can reverse order of loops to get rid of this traci call
                for veh_id in veh_list:
                    vehicle_data = scResults[veh_id]
                    accumulated_waiting_time = vehicle_data[tc.VAR_ACCUMULATED_WAITING_TIME]
                    if veh_id not in self.vehicles_on_incoming_lanes:
                        self.vehicles_on_incoming_lanes[veh_id] = {in_lane: vehicle_data[tc.VAR_WAITING_TIME]}
                    else:
                        self.vehicles_on_incoming_lanes[veh_id][in_lane] = accumulated_waiting_time - sum([self.vehicles_on_simulation[veh_id]["lanes"][lane] for lane in self.vehicles_on_simulation[veh_id]["lanes"].keys() if lane != in_lane])

    def check_if_vehicle_pass_the_junctions(self):
        for signal_id in self.signal_ids:
            for out_lane in self.signals[signal_id].outbound_lanes:
                veh_list = list(traci.lane.getLastStepVehicleIDs(out_lane))
                for veh_id in veh_list:
                    if veh_id not in self.vehicles_on_outcoming_lanes:
                        self.vehicles_on_outcoming_lanes[veh_id] = {
                            # get time of appearance on oncoming lane for veh_id
                            signal_id: {
                                out_lane: self.sim_step
                            }
                        }
                    else:
                        if signal_id not in self.vehicles_on_outcoming_lanes[veh_id].keys():
                            self.vehicles_on_outcoming_lanes[veh_id][signal_id] = {
                                out_lane: self.sim_step
                            }

    def get_total_queued_in_simulation(self):
        scResults = self.get_traci_subscription(self._all_vehicles_subscription)

        queue_list = []
        for id in self.lanes_and_junctions_ids:
            car_list = traci.lane.getLastStepVehicleIDs(id)
            for car_id in car_list:
                # new vehicles that appear in the simulation and have zero speed should not be considered in the queue
                vehicle_data = scResults[car_id]
                if vehicle_data[tc.VAR_SPEED] <= 0.1 and vehicle_data[tc.VAR_WAITING_TIME] != 0:
                    queue_list.append(car_id)
        # return sum([traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in self.lanes_and_junctions_ids] if
        # traci.lane)
        return len(queue_list)

    def get_total_waiting_time_in_simulation(self):
        return sum([int(traci.lane.getWaitingTime(lane_id)) for lane_id in self.lanes_and_junctions_ids])

    def calculate_current_delays_of_all_vehicles_in_simulation(self):
        scResults = self.get_traci_subscription(self._all_vehicles_subscription)

        delta_of_delays = []
        for veh_id in self.vehicles_on_simulation:
            if (
                    self.vehicles_on_simulation[veh_id]["time"]["time_of_appearance"]
                    != self.sim_step
                    and not self.vehicles_on_simulation[veh_id][
                "last_calculate_delta_of_delays"
            ]
            ):
                delta_of_delays.append(
                    self.vehicles_on_simulation[veh_id]["max_speed"]
                    * (
                        (
                                (
                                        self.sim_step
                                        - self.vehicles_on_simulation[veh_id]["time"][
                                            "time_of_appearance"
                                        ]
                                )
                                / (
                                    max(scResults[veh_id][tc.VAR_DISTANCE], 0.0001)
                                )
                        )
                        if "time_of_disappearance"
                           not in self.vehicles_on_simulation[veh_id]["time"]
                        else self.vehicles_on_simulation[veh_id]["time"][
                                 "time_of_total_journey"
                             ]
                             / self.routes_info[self.vehicles_on_simulation[veh_id]["routeID"]][
                                 "length"
                             ]
                    )
                )

        return delta_of_delays

    def calculate_delta_of_delays(self):
        scResults = self.get_traci_subscription(self._all_vehicles_subscription)


        delta_of_delays = []
        delta_time = self.step_length - self.yellow_length - self.red_length
        for veh_id in self.vehicles_on_simulation:
            if (
                    self.vehicles_on_simulation[veh_id]["time"]["time_of_appearance"]
                    != self.sim_step
                    and not self.vehicles_on_simulation[veh_id][
                "last_calculate_delta_of_delays"
            ]
            ):
                delta_of_delays.append(
                    self.vehicles_on_simulation[veh_id]["max_speed"]
                    * (
                            (
                                (
                                        (
                                                self.sim_step
                                                - self.vehicles_on_simulation[veh_id]["time"][
                                                    "time_of_appearance"
                                                ]
                                                - delta_time
                                        )
                                        / self.vehicles_on_simulation[veh_id]["total_distance"]
                                )
                                if self.vehicles_on_simulation[veh_id]["total_distance"] != 0
                                else 0
                            )
                            - (
                                (
                                        (
                                                self.sim_step
                                                - self.vehicles_on_simulation[veh_id]["time"][
                                                    "time_of_appearance"
                                                ]
                                        )
                                        / (
                                            max(scResults[veh_id][tc.VAR_DISTANCE], 0.0001)
                                        )
                                )
                                if "time_of_disappearance"
                                   not in self.vehicles_on_simulation[veh_id]["time"]
                                else self.vehicles_on_simulation[veh_id]["time"][
                                         "time_of_total_journey"
                                     ]
                                     / self.routes_info[
                                         self.vehicles_on_simulation[veh_id]["routeID"]
                                     ]["length"]
                            )
                    )
                )
                if (
                        "time_of_disappearance"
                        not in self.vehicles_on_simulation[veh_id]["time"].keys()
                ):
                    self.vehicles_on_simulation[veh_id][
                        "total_distance"
                    ] = scResults[veh_id][tc.VAR_DISTANCE]
                else:
                    self.vehicles_on_simulation[veh_id][
                        "last_calculate_delta_of_delays"
                    ] = True

        return mean(delta_of_delays) if len(delta_of_delays) != 0 else 0

    def get_number_of_vehicles_completing_journey(self):
        return len(
            [
                self.vehicles_on_simulation[veh]["time"]["time_of_total_journey"]
                for veh in self.vehicles_on_simulation
                if "time_of_total_journey" in self.vehicles_on_simulation[veh]["time"]
            ]
        )

    def get_total_time_of_journey(self):
        return sum(
            [
                self.vehicles_on_simulation[veh]["time"]["time_of_total_journey"]
                for veh in self.vehicles_on_simulation
                if "time_of_total_journey" in self.vehicles_on_simulation[veh]["time"]
            ]
        )

    def calculate_travel_time_and_delays(self):
        for veh_id in self.vehicles_on_simulation:
            if "time_of_total_journey" in self.vehicles_on_simulation[veh_id]["time"]:
                self.routes_info[self.vehicles_on_simulation[veh_id]["routeID"]][
                    "vehicle_type"
                ][self.vehicles_on_simulation[veh_id]["type"]]["real"]["vehicle_id"].append(
                    veh_id
                )
                self.routes_info[self.vehicles_on_simulation[veh_id]["routeID"]][
                    "vehicle_type"
                ][self.vehicles_on_simulation[veh_id]["type"]]["real"][
                    "total_travel_time"
                ].append(
                    self.vehicles_on_simulation[veh_id]["time"]["time_of_total_journey"]
                )
                self.routes_info[self.vehicles_on_simulation[veh_id]["routeID"]][
                    "total_travel_time_of_all_vehicles"
                ].append(
                    self.vehicles_on_simulation[veh_id]["time"]["time_of_total_journey"]
                )
                self.routes_info[self.vehicles_on_simulation[veh_id]["routeID"]][
                    "total_ideal_travel_time_of_all_vehicles"
                ].append(self.vehicles_on_simulation[veh_id]["ideal_travel_time"])

        for route_id in self.routes_info.keys():
            for veh_type in self.routes_info[route_id]["vehicle_type"]:
                self.routes_info[route_id]["vehicle_type"][veh_type]["real"][
                    "number_of_vehicles"
                ] = len(
                    self.routes_info[route_id]["vehicle_type"][veh_type]["real"][
                        "vehicle_id"
                    ]
                )
                self.routes_info[route_id]["vehicle_type"][veh_type]["real"][
                    "average_travel_time"
                ] = (
                    mean(
                        self.routes_info[route_id]["vehicle_type"][veh_type]["real"][
                            "total_travel_time"
                        ]
                    )
                    if len(
                        self.routes_info[route_id]["vehicle_type"][veh_type]["real"][
                            "vehicle_id"
                        ]
                    )
                       != 0
                    else 0
                )
                self.routes_info[route_id]["vehicle_type"][veh_type]["real"]["delays"][
                    "total"
                ] = [
                    delays
                    / self.routes_info[route_id]["vehicle_type"][veh_type]["ideal"][
                        "travel_time"
                    ]
                    for delays in self.routes_info[route_id]["vehicle_type"][veh_type][
                        "real"
                    ]["total_travel_time"]
                ]
                self.routes_info[route_id]["vehicle_type"][veh_type]["real"]["delays"][
                    "average"
                ] = (
                    mean(
                        self.routes_info[route_id]["vehicle_type"][veh_type]["real"][
                            "delays"
                        ]["total"]
                    )
                    if len(
                        self.routes_info[route_id]["vehicle_type"][veh_type]["real"][
                            "delays"
                        ]["total"]
                    )
                       != 0
                    else []
                )
                self.routes_info[route_id]["total_delays_of_all_vehicles"] = (
                        self.routes_info[route_id]["total_delays_of_all_vehicles"]
                        + self.routes_info[route_id]["vehicle_type"][veh_type]["real"][
                            "delays"
                        ]["total"]
                )

        for route_id in self.routes_info.keys():
            self.routes_info[route_id][
                "total_number_of_all_vehicles_generated-ThruPut_Scheduled"
            ] = sum(route_id in veh_id for veh_id in self.vehicles_on_simulation.keys())
            self.routes_info[route_id][
                "total_number_of_all_vehicles_completing_journey-ThruPut_Actual"
            ] = sum(
                self.routes_info[route_id]["vehicle_type"][veh_type]["real"][
                    "number_of_vehicles"
                ]
                for veh_type in self.routes_info[route_id]["vehicle_type"]
            )
            self.routes_info[route_id]["throughput_of_the_route-ThruPut_Idx"] = (
                self.routes_info[route_id][
                    "total_number_of_all_vehicles_completing_journey-ThruPut_Actual"
                ]
                / self.routes_info[route_id][
                    "total_number_of_all_vehicles_generated-ThruPut_Scheduled"
                ]
                if self.routes_info[route_id][
                       "total_number_of_all_vehicles_completing_journey-ThruPut_Actual"
                   ]
                   != 0
                else 0
            )
            self.routes_info[route_id]["total_average_travel_time_of_all_vehicles"] = (
                mean(self.routes_info[route_id]["total_travel_time_of_all_vehicles"])
                if len(self.routes_info[route_id]["total_travel_time_of_all_vehicles"]) != 0
                else 0
            )
            self.routes_info[route_id][
                "total_average_delays_of_all_vehicles-Delay_Idx_Average"
            ] = (
                mean(self.routes_info[route_id]["total_delays_of_all_vehicles"])
                if len(self.routes_info[route_id]["total_delays_of_all_vehicles"]) != 0
                else 0
            )
            self.routes_info[route_id]["Delay_Idx_StDev"] = (
                stdev(self.routes_info[route_id]["total_delays_of_all_vehicles"])
                if len(self.routes_info[route_id]["total_delays_of_all_vehicles"]) > 1
                else 0
            )
            self.routes_info[route_id][
                "total_average_delays_of_all_vehicles_with_weights"
            ] = (
                    len(self.routes_info[route_id]["total_travel_time_of_all_vehicles"])
                    / self.get_number_of_vehicles_completing_journey()
                    if self.get_number_of_vehicles_completing_journey() != 0
                    else 0
                ) * (
                    mean(self.routes_info[route_id]["total_delays_of_all_vehicles"])
                    if len(self.routes_info[route_id]["total_travel_time_of_all_vehicles"]) != 0
                    else 0
                )

        # calculate the total delay of all vehicles on all routes, which completed its journey
        # for index, delays in enumerate([self.routes[route_id]['total_delays_of_all_vehicles'] for route_id in self.routes.keys() if len(self.routes[route_id]['total_delays_of_all_vehicles'])]):
        #    self.total_delays_of_all_vehicles_from_all_routes.extend(delays)
        for route_id in self.routes_info.keys():
            if len(self.routes_info[route_id]["total_delays_of_all_vehicles"]) != 0:
                self.total_delays_of_all_vehicles_from_all_routes.extend(
                    self.routes_info[route_id]["total_delays_of_all_vehicles"]
                )
                self.total_real_travel_times_all_vehicles_from_all_routes.extend(
                    self.routes_info[route_id]["total_travel_time_of_all_vehicles"]
                )
                self.total_ideal_travel_times_all_vehicles_from_all_routes.extend(
                    self.routes_info[route_id]["total_ideal_travel_time_of_all_vehicles"]
                )

    def get_total_average_delays_with_weights(self):
        total_delays = []
        for route_id in self.routes_info.keys():
            if len(self.routes_info[route_id]["total_delays_of_all_vehicles"]) != 0:
                total_delays.append(
                    self.routes_info[route_id][
                        "total_average_delays_of_all_vehicles_with_weights"
                    ]
                )
        return sum(total_delays)

    def generate_training_file_with_routes(self):

        # load available routes in the environment
        with codecs.open(str(Path("environments", self.map_name, "routes", "default.yaml")), "r", "utf-8") as file:
            routes_descr = yaml.safe_load(file)
        # load the available types of vehicles in the environment
        with codecs.open(str(Path("environments", self.map_name, "vehicles", "default.yaml")), "r", "utf-8") as file:
            vehicles_descr = yaml.safe_load(file)

        path_to_save_rou = Path("environments", self.map_name, "training_routes_files_generated", self.run_name, "train.rou.xml")

        if not os.path.exists(str(path_to_save_rou)[:-14]):
            os.makedirs(str(path_to_save_rou)[:-14])

        # generate random new routes based on original data with routes from 26/11/2020 and 18/03/2021
        # if self.run % 2 == 0:
        path_from_original_rou = Path("environments", self.map_name,
                                      self.validation_day_directory_name,
                                      f"{self.validation_period_file_name}")
        # else:
        # path_from_original_rou = Path("environments", self.map_name", "websterCalculated", "18MarFull", "BB5B_7-8am.rou.xml")

        rou_generator = RoutesGenerator(
            path_to_save_rou=path_to_save_rou,
            routes=routes_descr,
            vehicles=vehicles_descr,
            force=True,
            begin=self.start_time,
            end=self.end_time,
            total_time=self.end_time - self.start_time
        )

        if rou_generator.path.exists():
            rou_generator.path.unlink()
        rou_generator.path = Path("environments", self.map_name, "training_routes_files_generated", self.run_name, f"train_{self.run}.rou.xml")
        rou_generator(path_from_original_rou=path_from_original_rou)

        train_file = rou_generator.path
        return train_file
