import os
import random
from itertools import combinations
import numpy as np
import pandas as pd
import time
from Genetic_Algorithm import GeneticAlgorithm


class AdaptationOptimizer:
    def __init__(self, max_generation, pop_size, mutation_rate, crossover_rate, compared_algorithms, system, optimization_goal):
        self.max_generation = max_generation
        self.pop_size = pop_size
        self.compared_algorithms = compared_algorithms
        self.system = system
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.optimization_goal = optimization_goal


    def dynamic_optimization(self, data_folder, data_files, run_no):

        # Store the initial seed to allow the comparison algorithm to use the same initial seeds
        initial_seeds = None
        initial_seeds_ids = None

        for selected_algorithm in self.compared_algorithms:

            self.his_pop_configs = []  # Storing optimized populations of each workload environments (config space)
            self.his_pop_perfs = []  # Storing optimized populations of each workload environments (perf space)
            self.his_pop_ids = []  # Storing optimized populations of each workload environments (corresponding ids)
            self.his_envs_name = []  # Storing workload environment name
            self.his_evaluated_configs_to_perfs = []  # Dic: Storing evaluated config -> perf of each workload environment
            self.similarity_score = {}  # Storing the similarity_score

            # Only LiDOS has unique environmental selection strategy
            if selected_algorithm == 'LiDOS':
                environmental_selection_type = 'LiDOS_selection'
            else:
                environmental_selection_type = 'Traditional_selection'

            output_folder_pop_perf = 'results/' + self.system + '/' + selected_algorithm + '/optimized_pop_perf_run_' + str(run_no)
            output_folder_pop_config = 'results/' + self.system + '/' + selected_algorithm + '/optimized_pop_config_run_' + str(run_no)
            if not os.path.exists(output_folder_pop_perf):
                os.makedirs(output_folder_pop_perf)
            if not os.path.exists(output_folder_pop_config):
                os.makedirs(output_folder_pop_config)

            # Iterate each workload and tuning system under corresponding workload environment
            for i, csv_file in enumerate(data_files):
                environment_name = os.path.splitext(csv_file)[0]
                data = pd.read_csv(os.path.join(data_folder, csv_file))
                config_space = data.iloc[:, :-1].values
                perf_space = data.iloc[:, -1].values
                print(f"Run{run_no}: {selected_algorithm} is running for the workload({environment_name}) under the system({self.system})")
                if i < 2:
                    self.similarity_score[environment_name] = 0

                # Record the running time of different algorithm
                time_start = time.time()

                if i == 0:
                    # For the first workload environment, the random population is initialized and saved
                    if initial_seeds is None:
                        initial_seeds, initial_seeds_ids = self.initialize_population(config_space, self.pop_size)
                    init_pop_config = initial_seeds
                    init_pop_config_ids = initial_seeds_ids
                else:
                    # For environments that are not first workload environments, different policies will make different response strategies
                    # to generate populations for the new workload environment
                    init_pop_config, init_pop_config_ids = self.generate_seeds_for_next_workload(config_space, selected_algorithm, environment_name)


                # Evolutionary algorithm is used to obtain superior config
                ga = GeneticAlgorithm(self.pop_size, self.mutation_rate, self.crossover_rate, self.optimization_goal)
                optimized_pop_configs, optimized_pop_perfs, optimized_pop_indices, evaluated_configs_to_perfs = ga.run(init_pop_config, init_pop_config_ids, config_space, perf_space, self.max_generation, environmental_selection_type, selected_algorithm, run_no, self.system, environment_name)
                print(f"Run0: The optimized perf of {selected_algorithm} is {optimized_pop_perfs.tolist()[1]}")

                time_end = time.time()
                time_sum = np.array([time_end - time_start])

                # Save the optimized results for experimental data analysis
                np.savetxt(os.path.join(output_folder_pop_config, f'{environment_name}_config.csv'), optimized_pop_configs, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(output_folder_pop_perf, f'{environment_name}_perf.csv'), optimized_pop_perfs, delimiter=',')
                np.savetxt(os.path.join(output_folder_pop_config, f'{environment_name}_indices.csv'), optimized_pop_indices, delimiter=',', fmt='%d')
                np.savetxt(os.path.join(output_folder_pop_perf, f'{environment_name}_time.csv'), time_sum, delimiter=',')

                # update or save related data, which will be reused in the subsequent optimization
                self.his_evaluated_configs_to_perfs.append(evaluated_configs_to_perfs)
                self.his_pop_configs.append(optimized_pop_configs)
                self.his_pop_perfs.append(optimized_pop_perfs)
                self.his_pop_ids.append(optimized_pop_indices)
                self.his_envs_name.append(environment_name)

            df = pd.DataFrame([self.similarity_score])
            df.to_csv(os.path.join(output_folder_pop_perf, 'similarity_score.csv'), index_label=False)



    def initialize_population(self, config_space, required_size, existing_configs=None, existing_ids=None):

        ''' As the existing dataset did not test all the config to obtain corresponding perf, simply generate configs may not exist in the dataset
        -50% from dataset, 50% from randomly generate to guarantee algorithm running
        -remark: it's necessary to keep each config in population is unique
        -all the compared algorithm should keep consistent to guarantee fair comparison
        '''

        existing_configs_hashes = set(map(lambda x: hash(x.tobytes()), existing_configs)) if existing_configs is not None else set()
        existing_ids = set(existing_ids) if existing_ids is not None else set()

        pop_size_from_data = required_size // 2
        pop_configs_from_data = []
        pop_ids_from_data = []

        while len(pop_configs_from_data) < pop_size_from_data:
            idx = np.random.choice(len(config_space))
            config = config_space[idx]
            config_hash = hash(config.tobytes())
            if idx not in existing_ids:
                pop_configs_from_data.append(config)
                pop_ids_from_data.append(idx)

                existing_configs_hashes.add(config_hash)
                existing_ids.add(idx)
        pop_configs_from_data = np.array(pop_configs_from_data)

        pop_configs_from_random = []
        while len(pop_configs_from_random) < required_size - pop_size_from_data:
            config = np.array([np.random.choice(np.unique(config_space[:, i])) for i in range(config_space.shape[1])])
            config_hash = hash(config.tobytes())
            if config_hash not in existing_configs_hashes:
                pop_configs_from_random.append(config)
                existing_configs_hashes.add(config_hash)
        population_configs_from_random = np.array(pop_configs_from_random)

        # Distribute id for randomly generated configs (do not exist in dataset -> -1)
        pop_ids_from_random = []
        for config in population_configs_from_random:
            matches = np.where((config_space == config).all(axis=1))[0]
            id = matches[0] if matches.size > 0 else -1
            pop_ids_from_random.append(id)

        population_configs = np.vstack((pop_configs_from_data, population_configs_from_random))
        population_ids = np.concatenate((pop_ids_from_data, pop_ids_from_random))
        return population_configs, population_ids

    def generate_seeds_for_next_workload(self, config_space, selected_algorithm, environment_name, beta=0.3):

        #  ************Stationary planner************
        if selected_algorithm == 'Vanilla-GA':
            init_pop_config, init_pop_config_ids = self.initialize_population(config_space, self.pop_size)

        #  ************Seed-EA (Dynamic adaptation)************
        elif selected_algorithm == 'Transfer-GA':
            init_pop_config_ids = self.his_pop_ids[-1]
            init_pop_config = self.his_pop_configs[-1]

        return init_pop_config, init_pop_config_ids










