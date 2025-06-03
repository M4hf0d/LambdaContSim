# LambdaFogSim Simulation Lifecycle: Detailed Function & File Map

Below is a comprehensive guide to the LambdaFogSim simulation lifecycle, tracing the execution flow through specific functions and files.

## 1. Initialization Phase

### Command-Line Argument Processing

- **File:** lambdafogsim.py
- **Functions:**
  - `main()` - Entry point for the simulator
  - `get_parser()` - Creates argument parser for command line options
  - `gc.init()` - Initializes global constants and paths

### Configuration Loading

- **File:** config.py
- **Functions:**
  - `parse_config(path)` - Loads and validates configuration from YAML
  - Sets global parameters:
    - `sim_num_of_epochs` - Total simulation duration
    - `sim_function_duration` - Execution time for functions
    - `infr_is_dynamic` - Whether infrastructure can change during simulation
    - `infr_node_crash_probability` - Likelihood of node failures
    - `infr_link_crash_probability` - Likelihood of link failures

### Infrastructure Setup

- **File:** lambdafogsim.py and utils.py
- **Functions:**
  - For physical infrastructure:
    - `generate_infrastructure(args.physical_infrastructure)` - Creates nodes and links from YAML
    - `dump_infrastructure(infrastructure, gc.SECF2F_INFRASTRUCTURE_PATH)` - Writes to Prolog file
  - For logical infrastructure:
    - `LogicalInfrastructure.loads(args.logical_infrastructure)` - Loads directly from Prolog file

## 2. Simulation Environment Setup

### SimPy Environment Creation

- **File:** lambdafogsim.py
- **Functions:**
  - `env = simpy.Environment()` - Creates SimPy environment
  - `env.process(simulation(env, config.sim_num_of_epochs, infrastructure))` - Registers main simulation process
  - Statistical collections initialization:
    - `applications_stats` - For tracking application placement/execution
    - `node_events` - For recording node failures/resurrections
    - `link_events` - For recording link failures/resurrections
    - `node_stats` - For tracking resource utilization

## 3. Main Simulation Loop

### Main Simulation Process

- **File:** lambdafogsim.py
- **Function:** `simulation(env, steps, infrastructure)`
  - Core simulation loop that runs for `steps` epochs

### For Each Epoch (Simulation Step)

1. **Infrastructure Dynamics Simulation**

   - `infrastructure.simulate_node_crash(category)` - Simulates node failures
   - `infrastructure.simulate_node_resurrection()` - Simulates node recovery
   - `infrastructure.simulate_link_crash()` - Simulates link failures
   - `infrastructure.simulate_link_resurrection()` - Simulates link recovery
   - `dump_infrastructure(infrastructure, gc.SECF2F_INFRASTRUCTURE_PATH)` - Updates Prolog file after changes

2. **Application Recovery Processing**

   - After failures, handle affected applications:
     - `get_starting_function()` - Identifies from where to restart
     - `get_recursive_dependents()` - Finds affected functions
     - `replace_application()` - Attempts to find new placements

3. **Event Generation**

   - For each event generator:
     - `take_decision(event_generator_trigger_probability)` - Decides if event should fire
     - `take_decision(event_probability)` - For each event in the generator

4. **Application Placement**

   - For each application triggered by events:
     - `place_application(application_name, config_application, generator_id, infrastructure, step_number)` - Tries to place the application
     - Within `place_application()`:
       - `get_raw_placement()` - Calls Prolog engine for placement decision (in `src/placement.py`)
       - `parse_placement()` - Processes the Prolog result

5. **Function Execution**

   - For each placed application:
     - `get_ready_functions(application_obj.chain)` - Identifies functions ready to execute
     - `FunctionProcess(function, env, application_obj)` - Creates SimPy process for each function

6. **Resource Monitoring**
   - For each node:
     - `node.get_load()` - Records current load
     - `node_stats[node.id][step_number] = node_data` - Stores in statistics
   - `yield env.timeout(1)` - Advances simulation time by one epoch

## 4. Function Execution Flow

### Function Process Execution

- **File:** function_process.py
- **Class:** `FunctionProcess`
- **Methods:**
  - `__init__(fun, env, application)` - Sets up function process
  - `run()` - SimPy process that simulates the function execution:
    - `yield self.env.timeout(config.sim_function_duration)` - Simulates execution time
    - Updates function state: `FunctionState.RUNNING` → `FunctionState.COMPLETED`
    - `get_direct_dependents()` - Finds functions that depend on this one
    - `delete_executed_function()` - Updates application chain
    - `get_ready_functions()` - Finds newly ready functions

## 5. Application Replacement

### Handling Failures

- **File:** lambdafogsim.py
- **Functions:**
  - `replace_application(application_obj, start_function, starting_nodes, crashed_nodes, crashed_link, infrastructure, step_number)` - Finds new placements for affected parts
  - Within replacement:
    - `get_raw_placement()` - With `PlacementType.REPLACEMENT` flag
    - For each successfully replaced function:
      - Updates the application chain
      - Creates new `FunctionProcess` instances

## 6. Termination and Reporting

### End of Simulation

- **File:** lambdafogsim.py
- **Functions:**
  - `env.run(until=config.sim_num_of_epochs)` - Runs until specified epochs
  - `print_stats()` - Calculates and prints simulation statistics
    - Processes `applications_stats`, `node_events`, etc.
    - Writes JSON report to `config.sim_report_output_file`
  - `export_csv_data()` - Exports detailed data to CSV files in `exports/` directory
  - `visualize_results()` - Calls visualization script

### Visualization

- **File:** vis.py
- **Functions:**
  - `main()` - Entry point for visualization
  - `load_report()` - Loads simulation data
  - `plot_placement_statistics()`, `plot_node_crash_events()`, etc. - Create various plots
  - Creates HTML report in visualization_results directory

### Data Export

- **File:** export_data.py
- **Functions:**
  - `export_all_data()` - Coordinates all exports
  - `export_placement_data()`, `export_node_data()`, etc. - Create CSV files

## Complete Flow Sequence

1. **Parse command-line arguments** → **Load configuration** → **Create infrastructure**
2. **Create SimPy environment** → **Register simulation process**
3. For each epoch:
   - **Simulate infrastructure failures/recoveries**
   - **Process application recovery**
   - **Generate events**
   - **Place applications**
   - **Execute ready functions**
   - **Monitor resources**
4. **Calculate statistics** → **Generate reports** → **Visualize results**

This detailed breakdown provides a comprehensive roadmap of the simulation execution flow through the LambdaFogSim code base.
