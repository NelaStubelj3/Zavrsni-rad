# Copyright 2023, by Julien Cegarra & Benoît Valéry. All rights reserved.
# Institut National Universitaire Champollion (Albi, France).
# License : CeCILL, version 2.1 (see the LICENSE file)
import configparser
import gettext
import sys
from pathlib import Path
from collections import namedtuple
from time import perf_counter
from datetime import datetime
from csv import DictWriter
from core.constants import PATHS, REPLAY_MODE
from core.utils import find_the_first_available_session_number, find_the_last_session_number

config = configparser.ConfigParser()
config.read('config.ini')
scenario_path = config['Openmatb']['scenario_path']

def get_scenario_prefix(scenario_path):
    return scenario_path.split('_')[1]  # Extracting 'easy', 'medium', or 'hard'

scenario_prefix = get_scenario_prefix(scenario_path)

class Logger:
    def __init__(self):
        self.datetime = datetime.now()
        self.fields_list = ['logtime', 'scenario_time', 'type', 'module', 'address', 'value']
        self.slot = namedtuple('Row', self.fields_list)
        self.maxfloats = 6  # Time logged at microsecond precision
        self.session_id = None
        self.lsl = None

        
        self.session_id = find_the_first_available_session_number()
        session_dir = PATHS['SESSIONS'].joinpath(self.datetime.strftime("%Y-%m-%d"), f"Session {self.session_id} {scenario_prefix}")
        self.path = session_dir.joinpath(f'{self.session_id}_{self.datetime.strftime("%y%m%d_%H%M%S")}.csv')
        self.path_sysmon = session_dir.joinpath(f'{self.session_id}_{self.datetime.strftime("%y%m%d_%H%M%S")}_sysmon.csv')
        self.path_track = session_dir.joinpath(f'{self.session_id}_{self.datetime.strftime("%y%m%d_%H%M%S")}_track.csv')
        self.path_track_input_x = session_dir.joinpath(f'{self.session_id}_{self.datetime.strftime("%y%m%d_%H%M%S")}_track_input_x.csv')
        self.path_track_input_y = session_dir.joinpath(f'{self.session_id}_{self.datetime.strftime("%y%m%d_%H%M%S")}_track_input_y.csv')
        self.path_track_state_x = session_dir.joinpath(f'{self.session_id}_{self.datetime.strftime("%y%m%d_%H%M%S")}_track_state_x.csv')
        self.path_track_state_y = session_dir.joinpath(f'{self.session_id}_{self.datetime.strftime("%y%m%d_%H%M%S")}_track_state_y.csv')
        self.path_resman = session_dir.joinpath(f'{self.session_id}_{self.datetime.strftime("%y%m%d_%H%M%S")}_resman.csv')
        self.path_scheduling = session_dir.joinpath(f'{self.session_id}_{self.datetime.strftime("%y%m%d_%H%M%S")}_scheduling.csv')
        self.path_performance = session_dir.joinpath(f'{self.session_id}_{self.datetime.strftime("%y%m%d_%H%M%S")}_performance.csv')

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = 'w'

        self.scenario_time = 0  # Updated by the scheduler class

        self.file = None
        self.file_sysmon = None
        self.file_track = None
        self.file_track_input_x = None
        self.file_track_input_y = None
        self.file_track_state_x = None
        self.file_track_state_y = None
        self.file_resman = None
        self.file_scheduling = None
        self.file_performance = None
        self.writer = None
        self.queue = list()

        if not REPLAY_MODE:
            self.open()


    # TODO: see if we can/should merge record_* methods into one
    def record_event(self, event):
        if len(event.command) == 1:
            adress = 'self'
            value = event.command[0]
        elif len(event.command) == 2:
            adress = event.command[0]
            value = event.command[1]
        slot = [perf_counter(), self.scenario_time, 'event', event.plugin, adress, value]
        self.write_single_slot(slot)


    def record_input(self, module, key, state):
        slot = [perf_counter(), self.scenario_time, 'input', module, key, state]
        self.write_single_slot(slot)


    def record_aoi(self, container, name):
        plugin = name.split('_')[0]
        widget = '_'.join(name.split('_')[1:])
        slot = [perf_counter(), self.scenario_time, 'aoi', plugin, widget, container.get_x1y1x2y2()]
        self.write_single_slot(slot)


    def record_state(self, graph_name, attribute, value):
        module = graph_name.split('_')[0]
        graph_name = '_'.join(graph_name.split('_')[1:])
        address = f'{graph_name}, {attribute}'
        slot = [perf_counter(), self.scenario_time, 'state', module, address, value]
        self.write_single_slot(slot)


    def record_parameter(self, plugin, address, value):
        slot = [perf_counter(), self.scenario_time, 'parameter', plugin, address, value]
        self.write_single_slot(slot)


    def log_performance(self, module, metric, value):
        slot = [perf_counter(), self.scenario_time, 'performance', module, metric, value]
        self.write_single_slot(slot)


    def record_a_pseudorandom_value(self, module, seed, output):
        slot = [perf_counter(), self.scenario_time, 'seed_value', module, '', seed]
        self.write_single_slot(slot)
        slot = [perf_counter(), self.scenario_time, 'seed_output', module, '', output]
        self.write_single_slot(slot)


    def log_manual_entry(self, entry, key='manual'):
        slot = [perf_counter(), self.scenario_time, key, '', '', entry]
        self.write_single_slot(slot)


    def __enter__(self):
        self.open()
        return self


    def __exit__(self, type, value, traceback):
        self.file.close()


    def open(self):
        create_header = False if self.path.exists() and self.mode == 'a' else True
        self.file = open(str(self.path), self.mode, newline = '')

        create_header = False if self.path_sysmon.exists() and self.mode == 'a' else True
        self.file_sysmon = open(str(self.path_sysmon), self.mode, newline = '')

        create_header = False if self.path_track.exists() and self.mode == 'a' else True
        self.file_track = open(str(self.path_track), self.mode, newline = '')

        create_header = False if self.path_track_input_x.exists() and self.mode == 'a' else True
        self.file_track_input_x = open(str(self.path_track_input_x), self.mode, newline = '')
        create_header = False if self.path_track_input_y.exists() and self.mode == 'a' else True
        self.file_track_input_y = open(str(self.path_track_input_y), self.mode, newline = '')

        create_header = False if self.path_track_state_x.exists() and self.mode == 'a' else True
        self.file_track_state_x = open(str(self.path_track_state_x), self.mode, newline = '')
        create_header = False if self.path_track_state_y.exists() and self.mode == 'a' else True
        self.file_track_state_y = open(str(self.path_track_state_y), self.mode, newline = '')
        
        create_header = False if self.path_resman.exists() and self.mode == 'a' else True
        self.file_resman = open(str(self.path_resman), self.mode, newline = '')

        create_header = False if self.path_scheduling.exists() and self.mode == 'a' else True
        self.file_scheduling = open(str(self.path_scheduling), self.mode, newline = '')

        create_header = False if self.path_performance.exists() and self.mode == 'a' else True
        self.file_performance = open(str(self.path_performance), self.mode, newline = '')

        self.writer = DictWriter(self.file, fieldnames=self.fields_list)

        if create_header:
            self.writer.writeheader()


    def close(self):
        self.file.close()


    def add_row_to_queue(self, row):
        self.queue.append(row)


    def empty_queue(self):
        self.queue = list()


    def round_row(self, row):
        new_list = list()
        for col in row:
            new_value = round(col, self.maxfloats) if isinstance(col, float) or isinstance(col, int) else col
            new_list.append(new_value)
        return self.slot(*new_list)


    def write_row_queue(self, change_dict=None):
        if not REPLAY_MODE:
            if len(self.queue) == 0:
                print(_('Warning, queue is empty'))
            else:
                for this_row in self.queue:
                    row_dict = self.round_row(this_row)._asdict()
                    if change_dict is not None:
                        for k,v in change_dict.items():
                            row_dict[k] = v

                    if this_row.module == 'sysmon':
                        self.writer = DictWriter(self.file_sysmon, fieldnames=self.fields_list)
                        self.writer.writerow(row_dict)
                    elif this_row.module == 'track':
                        self.writer = DictWriter(self.file_track, fieldnames=self.fields_list)
                        self.writer.writerow(row_dict)
                        if this_row.type == 'input':
                            if this_row.address == 'joystick_x':
                                self.writer = DictWriter(self.file_track_input_x, fieldnames=self.fields_list)
                                self.writer.writerow(row_dict)
                            if this_row.address == 'joystick_y':
                                self.writer = DictWriter(self.file_track_input_y, fieldnames=self.fields_list)
                                self.writer.writerow(row_dict)
                        if this_row.type == 'state':
                            if this_row.address == 'reticle, cursor_proportional_x':
                                self.writer = DictWriter(self.file_track_state_x, fieldnames=self.fields_list)
                                self.writer.writerow(row_dict)
                            if this_row.address == 'reticle, cursor_proportional_y':
                                self.writer = DictWriter(self.file_track_state_y, fieldnames=self.fields_list)
                                self.writer.writerow(row_dict)
                    elif this_row.module == 'resman':
                        self.writer = DictWriter(self.file_resman, fieldnames=self.fields_list)
                        self.writer.writerow(row_dict)
                    elif this_row.module == 'scheduling':
                        self.writer = DictWriter(self.file_scheduling, fieldnames=self.fields_list)
                        self.writer.writerow(row_dict)
                    elif this_row.module == 'performance':
                        self.writer = DictWriter(self.file_performance, fieldnames=self.fields_list)
                        self.writer.writerow(row_dict)

                    self.writer = DictWriter(self.file, fieldnames=self.fields_list)
                    self.writer.writerow(row_dict)

                    if self.lsl is not None:
                        self.lsl.push(';'.join([str(r) for r in row_dict.values()]))
                self.empty_queue()


    def write_single_slot(self, values):
        row = self.slot(*values)
        self.add_row_to_queue(row)
        self.write_row_queue()


    def set_totaltime(self, totaltime):
        self.totaltime = totaltime


    def set_scenario_time(self, scenario_time):
        self.scenario_time = scenario_time
        

logger = Logger()
