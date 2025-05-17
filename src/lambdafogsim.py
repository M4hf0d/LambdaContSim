import argparse
import json
from application.application import Application, ApplicationState
from application.placed_function import FunctionState
from config import parse_config
from infrastructure.utils import generate_infrastructure, dump_infrastructure
from infrastructure.infrastructure import Infrastructure, LinkInfo
from infrastructure.logical_infrastructure import LogicalInfrastructure
from infrastructure.node import NodeCategory
import os
import shutil
import config
from infrastructure.physical_infrastructure import PhysicalInfrastructure
from orchestration.function_process import FunctionProcess
import logs
from placement import (
    PlacementType,
    build_app_chain,
    get_raw_placement,
    parse_placement,
)
import simpy
import global_constants as gc
from utils import (
    get_ready_functions,
    get_recursive_dependents,
    get_recursive_waiting_dependencies,
    is_edge_part,
    take_decision,
    get_starting_function,
)
import random

# Statistical variables

applications_stats: dict = {}
node_events: list = []
node_stats: dict[str, dict] = {}
link_events: list = []


def get_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Simulation with LambdaFogSim")

    parser.add_argument(
        "-v",
        "--verbose",
        help="increase simulator verbosity",
        action="store_true",
        dest="verbose",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=gc.DEFAULT_CONFIG_PATH,
        help="simulator config (default config.yaml)",
        dest="config",
    )

    # Physical and logical infrastructure are mutually exclusive
    action = parser.add_mutually_exclusive_group()

    action.add_argument(
        "-p",
        "--physical",
        type=str,
        default=gc.DEFAULT_INFRASTRUCTURE_CONFIG_PATH,
        help="physical infrastructure config (default infrastructure_config.yaml)",
        dest="physical_infrastructure",
    )

    action.add_argument(
        "-l",
        "--logical",
        type=str,
        default=None,
        help="logical infrastructure Prolog definition",
        dest="logical_infrastructure",
    )

    return parser


def place_application(
    application_name: str,
    config_application: dict,
    generator_id: str,
    infrastructure: Infrastructure,
    epoch: int,
):

    # get the logger
    logger = logs.get_logger()

    # application should be padded?
    placement_type = PlacementType.PADDED_PLACEMENT
    if not config.sim_use_padding:
        placement_type = PlacementType.UNPADDED_PLACEMENT

    # try to place this app with SecFaaS2Fog

    # it will reply with a valid placement iff application can be placed
    raw_placement, execution_time = get_raw_placement(
        placement_type=placement_type,
        orchestration_id=config.applications[application_name]["orchestration_id"],
        generator_id=generator_id,
    )

    # get the node where the generator is placed
    generator_node = infrastructure.event_generators[generator_id].source_node

    application_can_be_placed = False

    if raw_placement is None:
        logger.critical("Query is None")
    elif raw_placement == {}:
        logger.info("Placement failed for %s", application_name)
    else:
        application_chain = build_app_chain(raw_placement)
        placement = parse_placement(raw_placement)

        # calculate for each function its dependency nodes
        functions = placement.keys()
        for function_name in functions:
            placed_function = placement[function_name]
            dependencies = application_chain[function_name]
            if len(dependencies) == 0:
                # starting function
                placed_function.previous_nodes.append(generator_node)
            else:
                for dependency in dependencies:
                    node_id = placement[dependency].node_id
                    if node_id not in placed_function.previous_nodes:
                        placed_function.previous_nodes.append(node_id)

        # print(placement)
        # print(application_chain)

        application_can_be_placed = True

        logger.info("Placement found for %s", application_name)

    if application_name not in applications_stats.keys():
        applications_stats[application_name] = {}
        applications_stats[application_name]["placements"] = {}
        applications_stats[application_name]["placements"]["data"] = []
        applications_stats[application_name]["replacements"] = {}
        applications_stats[application_name]["replacements"]["data"] = []

    # normal placement
    placement_data: dict = {
        "node": generator_node,
        "duration": execution_time,
        "epoch": epoch,
        "success": application_can_be_placed,
    }

    applications_stats[application_name]["placements"]["data"].append(placement_data)

    if application_can_be_placed:

        # create application instance
        application_obj = Application(
            application_name,
            config_application["filename"],
            config_application["orchestration_id"],
            epoch,
            application_chain,
            placement,
            infrastructure.nodes,
        )

        # place application over the infrastructure (update nodes capabilities: memory, number of vCPUs )
        functions = placement.keys()
        for function_id in functions:
            node_id = placement[function_id].node_id
            node_obj = infrastructure.nodes[node_id]

            # take resources from the node
            node_obj.take_resources(
                memory=placement[function_id].memory, v_cpu=placement[function_id].v_cpu
            )

        return application_obj

    return None


def replace_application(
    application_obj: Application,
    starting_function: str,
    starting_nodes: list[str],
    crashed_nodes: list,
    crashed_link: list,
    infrastructure: Infrastructure,
    epoch: int,
):
    # get the logger
    logger = logs.get_logger()

    # application should be padded?
    placement_type = PlacementType.PADDED_REPLACEMENT
    if not config.sim_use_padding:
        placement_type = PlacementType.UNPADDED_REPLACEMENT

    # get application name
    application_name = application_obj.name

    # try to place this app with SecFaaS2Fog

    # it will reply with a valid placement iff application can be placed
    raw_placement, execution_time = get_raw_placement(
        placement_type=placement_type,
        starting_function=starting_function,
        starting_nodes=starting_nodes,
        orchestration_id=config.applications[application_name]["orchestration_id"],
    )

    # if the excution time is much more (for us, 2 times more) than maximum allowed time it can be a problem
    if execution_time > config.sim_max_placement_time * 2000:
        logger.error(
            f"Replacement of application {application_obj.id} took {execution_time} milliseconds"
        )

    application_can_be_placed = False

    if raw_placement is None:
        logger.critical("Query is None")
    elif raw_placement == {}:
        logger.info("Replacement failed for %s", application_obj.id)
    else:
        new_application_chain = build_app_chain(raw_placement)
        new_placement = parse_placement(raw_placement)

        # print(new_placement)
        # print(new_application_chain)

        # calculate for each function its dependency nodes
        functions = new_placement.keys()
        for function_name in functions:
            new_placed_function = new_placement[function_name]
            new_function_dependencies = new_application_chain[function_name]
            if len(new_function_dependencies) == 0:
                # start function
                new_placed_function.previous_nodes += starting_nodes
            else:
                for dependency in new_function_dependencies:

                    node_id = new_placement[dependency].node_id

                    if node_id not in new_placed_function.previous_nodes:
                        new_placed_function.previous_nodes.append(node_id)

        application_can_be_placed = True

        logger.info("Replacement found for %s", application_obj.id)

    #  replacement
    placement_data: dict = {
        "crashed_nodes": crashed_nodes,
        "crashed_links": crashed_link,
        "duration": execution_time,
        "epoch": epoch,
        "success": application_can_be_placed,
    }

    applications_stats[application_name]["replacements"]["data"].append(placement_data)

    if application_can_be_placed:

        # update the application replacing old placed functions with new ones
        functions = new_placement.keys()
        for function_id in functions:
            application_obj.placement[function_id] = new_placement[function_id]

        # place new functions over the infrastructure (update nodes capabilities: memory, number of vCPUs )
        for function_id in functions:
            node_id = new_placement[function_id].node_id
            node_obj = infrastructure.nodes[node_id]

            # take resources from the node
            node_obj.take_resources(
                memory=new_placement[function_id].memory,
                v_cpu=new_placement[function_id].v_cpu,
            )

        return True, new_application_chain

    return False, None


def simulation(env: simpy.Environment, steps: int, infrastructure: Infrastructure):

    # get logger
    logger = logs.get_logger()

    # list of applications
    list_of_applications: list[Application] = []

    for step_number in range(0, steps):

        # print the step
        logger.info("--- STEP %d ---", step_number)

        # CRASH/RESURRECTION PHASE

        # reset variables
        crashed_node_id = {}
        resurrected_node_id = {}
        crashed_link: tuple = None
        resurrected_link: tuple = None
        link_resurr_first_node = None
        link_resurr_second_node = None
        link_crashed_first_node = None
        link_crashed_second_node = None

        # True iff a node or a link crashed
        crash_occurred = False

        # True iff a node or a link resurrected
        resurrection_occurred = False

        # NODE crash
        for category in ["cloud", "fog", "edge"]:

            node_could_crash = take_decision(
                config.infr_node_crash_probability[category]
            )

            node_could_resurrect = take_decision(
                config.infr_node_resurrection_probability[category]
            )

            crashed_node_id[category] = None
            resurrected_node_id[category] = None

            if node_could_crash:
                crashed_node_id[category] = infrastructure.simulate_node_crash(
                    NodeCategory.from_string(category)
                )

                if crashed_node_id[category] is not None:
                    logger.info("Node %s crashed", crashed_node_id[category])

                    # a node crashed
                    crash_occurred = True

                    # add event to the list
                    event = {
                        "type": "crash",
                        "node_id": crashed_node_id[category],
                        "epoch": step_number,
                    }
                    node_events.append(event)

            if node_could_resurrect:
                if crashed_node_id[category] is not None:
                    # node which just crashed can't resurrect in the same epoch
                    resurrected_node_id[
                        category
                    ] = infrastructure.simulate_node_resurrection(
                        NodeCategory.from_string(category), crashed_node_id[category]
                    )
                else:
                    resurrected_node_id[
                        category
                    ] = infrastructure.simulate_node_resurrection(
                        NodeCategory.from_string(category)
                    )

                if resurrected_node_id[category] is not None:
                    logger.info("Node %s resurrected", resurrected_node_id[category])

                    # a node resurrected
                    resurrection_occurred = True

                    # add event to the list
                    event = {
                        "type": "resurrection",
                        "node_id": resurrected_node_id[category],
                        "epoch": step_number,
                    }
                    node_events.append(event)

        # clean crashed nodes dict deleting Null references and trasform it into a list
        crashed_nodes = [node for node in list(crashed_node_id.values()) if node]
        # clean resurrected nodes dict deleting Null references and trasform it into a list
        resurrected_nodes = [
            node for node in list(resurrected_node_id.values()) if node
        ]

        # LINK crash
        link_could_crash = take_decision(config.infr_link_crash_probability)

        link_could_resurrect = take_decision(config.infr_link_resurrection_probability)

        if link_could_crash:
            (
                link_crashed_first_node,
                link_crashed_second_node,
            ) = infrastructure.simulate_link_crash()

            if (
                link_crashed_first_node is not None
                and link_crashed_second_node is not None
            ):
                logger.info(
                    "Link %s <-> %s crashed",
                    link_crashed_first_node,
                    link_crashed_second_node,
                )

                # a link crashed
                crash_occurred = True

                # add event to the list
                event = {
                    "type": "crash",
                    "first_node_id": link_crashed_first_node,
                    "second_node_id": link_crashed_second_node,
                    "epoch": step_number,
                }
                link_events.append(event)

                crashed_link = (link_crashed_first_node, link_crashed_second_node)

        if link_could_resurrect:

            if crashed_link is not None:
                # link which just crashed can't resurrect in the same epoch
                (
                    link_resurr_first_node,
                    link_resurr_second_node,
                ) = infrastructure.simulate_link_resurrection(crashed_link)
            else:
                (
                    link_resurr_first_node,
                    link_resurr_second_node,
                ) = infrastructure.simulate_link_resurrection()

            if (
                link_resurr_first_node is not None
                and link_resurr_second_node is not None
            ):
                logger.info(
                    "Link %s <-> %s resurrected",
                    link_resurr_first_node,
                    link_resurr_second_node,
                )

                # a link resurrected
                resurrection_occurred = True

                # add event to the list
                event = {
                    "type": "resurrection",
                    "first_node_id": link_resurr_first_node,
                    "second_node_id": link_resurr_second_node,
                    "epoch": step_number,
                }
                link_events.append(event)

                resurrected_link = (link_resurr_first_node, link_resurr_second_node)

        # needs infrastructure.pl to be updated?
        infrastructure_needs_update = crash_occurred or resurrection_occurred
        if infrastructure_needs_update:

            if isinstance(infrastructure, PhysicalInfrastructure):

                # if something resurrected recalculate links which had been affected by past crashes
                if resurrection_occurred:
                    infrastructure.recalculate_routes_after_resurrection(
                        resurrected_nodes, resurrected_link
                    )

                # if something crashed recalculate only necessary paths (those which are involved by node/link crash)
                if crash_occurred:
                    infrastructure.recalculate_routes_after_crash(
                        crashed_nodes, crashed_link
                    )

            # nodes or links crashed/resurrected - update infastructure.pl
            logger.info("Infrastructure changes - update infrastructure")
            dump_infrastructure(infrastructure, gc.SECF2F_INFRASTRUCTURE_PATH)

        # somehing crashed?
        if crash_occurred:
            # there are affected applications?
            for application_obj in list_of_applications:

                # completed or canceled apps are not interesting
                if application_obj.state in [
                    ApplicationState.COMPLETED,
                    ApplicationState.CANCELED,
                ]:
                    continue

                # list of functions interested by the crash
                interested_functions = set()

                # find functions interested by the crash
                for placed_function in application_obj.placement.values():

                    # NODE crash
                    # check only waiting and running functions
                    # Option 1:
                    # check if there is at least one function deployed in a crashed node
                    # OR
                    # Option 2:
                    # check if there is at least one function which depends on a service deployed in a crashed node

                    # function is interested by the node crash?
                    node_crash_verified = False

                    if crashed_nodes:

                        if placed_function.state in [
                            FunctionState.WAITING,
                            FunctionState.RUNNING,
                        ]:
                            # Option 1: is the function deployed in a crashed node?
                            if placed_function.node_id in crashed_nodes:
                                interested_functions.add(placed_function.id)

                                # if the function is waiting
                                if placed_function.state == FunctionState.WAITING:

                                    # also its dependency functions are interested
                                    result = set()
                                    get_recursive_waiting_dependencies(
                                        placed_function.id,
                                        application_obj.chain,
                                        application_obj.placement,
                                        result,
                                    )
                                    for function_id in result:
                                        interested_functions.add(function_id)

                                node_crash_verified = True
                            else:
                                # Option 2: is one of its services deployed in a crashed node?
                                for service_id in placed_function.linked_services:
                                    service = infrastructure.services[service_id]
                                    if service.deployed_node in crashed_nodes:
                                        interested_functions.add(placed_function.id)
                                        
                                        # if the function is waiting
                                        if placed_function.state == FunctionState.WAITING:

                                            # also its dependency functions are interested
                                            result = set()
                                            get_recursive_waiting_dependencies(
                                                placed_function.id,
                                                application_obj.chain,
                                                application_obj.placement,
                                                result,
                                            )
                                            for function_id in result:
                                                interested_functions.add(function_id)

                                        node_crash_verified = True
                                        break

                    # LINK crash
                    # Option 1: Crashed link between 2 functions
                    # if there is a waiting function2 deployed on nodeY
                    # for any function1 deployed on nodeX
                    # where function2 depends on function1
                    # check if the link (node1, node2) belongs to the path from nodeX to nodeY
                    # OR
                    # Option 2: Crashed link between function and one of its service
                    # if there is a waiting or running functionX on nodeX
                    # for any serviceY deployed on nodeY
                    # where functionX depends on serviceY
                    # check if the link (node1, node2) belongs to the path from nodeX to nodeY

                    # don't check if the function is already 'interested'
                    if not node_crash_verified and crashed_link:

                        option1_verified = False

                        # Option 1
                        if placed_function.state == FunctionState.WAITING:
                            node_y = placed_function.node_id
                            previous_nodes = placed_function.previous_nodes
                            for node_x in previous_nodes:
                                path = infrastructure.links[node_x][LinkInfo.PATH][
                                    node_y
                                ]
                                if is_edge_part(
                                    path,
                                    link_crashed_first_node,
                                    link_crashed_second_node,
                                ):
                                    interested_functions.add(placed_function.id)
                                    
                                    # also its dependency functions are interested
                                    result = set()
                                    get_recursive_waiting_dependencies(
                                        placed_function.id,
                                        application_obj.chain,
                                        application_obj.placement,
                                        result,
                                    )
                                    for function_id in result:
                                        interested_functions.add(function_id)
                                    
                                    option1_verified = True

                                    break

                        # Option 2
                        # check only if option 1 is not verified
                        if not option1_verified and placed_function.state in [
                            FunctionState.WAITING,
                            FunctionState.RUNNING,
                        ]:
                            node_x = placed_function.node_id
                            for service_id in placed_function.linked_services:
                                service_node_y = infrastructure.services[
                                    service_id
                                ].deployed_node
                                path = infrastructure.links[node_x][LinkInfo.PATH][
                                    service_node_y
                                ]
                                if is_edge_part(
                                    path,
                                    link_crashed_first_node,
                                    link_crashed_second_node,
                                ):
                                    interested_functions.add(placed_function.id)

                                    # if the function is waiting
                                    if placed_function.state == FunctionState.WAITING:

                                        # also its dependency functions are interested
                                        result = set()
                                        get_recursive_waiting_dependencies(
                                            placed_function.id,
                                            application_obj.chain,
                                            application_obj.placement,
                                            result,
                                        )
                                        for function_id in result:
                                            interested_functions.add(function_id)

                                    break

                # if there is interested functions then application needs to be partially replaced
                if interested_functions:
                    logger.info(
                        "Application %s needs a partial replacement", application_obj.id
                    )

                    # find the starting function
                    start_function = get_starting_function(
                        list(interested_functions),
                        application_obj.original_chain,
                        application_obj.placement
                    )

                    # get the list of function which depends by start_function
                    dependent_functions = get_recursive_dependents(
                        start_function, application_obj.original_chain
                    )

                    functions_to_be_released = [start_function] + dependent_functions

                    # release all resources of start function and its dependents, if they are running or waiting
                    for function_name in functions_to_be_released:
                        function = application_obj.placement[function_name]

                        if function.state not in [FunctionState.RUNNING, FunctionState.WAITING]:
                            continue

                        node_id = function.node_id
                        node_obj = application_obj.infrastructure_nodes[node_id]
                        node_obj.release_resources(function.memory, function.v_cpu)

                        logger.info(
                            "Application %s - Function %s has been canceled",
                            application_obj.id,
                            function.id,
                        )

                        # interrupt the function if it is running
                        if function.state == FunctionState.RUNNING:

                            # set function as interrupted
                            function.state = FunctionState.INTERRUPTED

                            application_obj.function_processes[
                                function.id
                            ].action.interrupt()

                        else:
                            # set function state as canceled
                            function.state = FunctionState.CANCELED

                    # get starting nodes for replacement
                    starting_nodes = application_obj.placement[
                        start_function
                    ].previous_nodes

                    # application can't be replaced if almost one of the starting nodes is down
                    replacement_can_start = True

                    for node_id in starting_nodes:
                        if node_id in infrastructure.crashed_nodes:
                            logger.info(
                                "Replacement will not start because one of the starting node is crashed"
                            )
                            replacement_can_start = False
                            break

                    if not replacement_can_start:
                        (
                            is_successfully_replaced,
                            replaced_application_chain,
                        ) = False, None

                    else:
                        # search for a new placement

                        # save application file into default path
                        application_path = os.path.join(
                            gc.APPLICATIONS_DIR, application_obj.filename
                        )
                        shutil.copy(application_path, gc.SECF2F_APP_PATH)

                        (
                            is_successfully_replaced,
                            replaced_application_chain,
                        ) = replace_application(
                            application_obj,
                            start_function,
                            starting_nodes,
                            crashed_nodes,
                            list(crashed_link) if crashed_link else [],
                            infrastructure,
                            step_number,
                        )

                    if is_successfully_replaced:
                        # update the application chain with the new placement
                        functions = replaced_application_chain.keys()
                        for function in functions:
                            
                            # already completed functions don't have the key inside the chain
                            if function not in application_obj.chain:
                                application_obj.chain[function] = []
                            
                            for dependent_function in replaced_application_chain[
                                function
                            ]:
                                if (
                                    dependent_function
                                    not in application_obj.chain[function]
                                ):
                                    application_obj.chain[function].append(
                                        dependent_function
                                    )

                        # start the replaced section of the application
                        # if it is the right moment (previus functions in the chain finished)
                        ready_replaced_functions = get_ready_functions(
                            replaced_application_chain
                        )
                        ready_functions = get_ready_functions(application_obj.chain)
                        # execute ready functions
                        for function_name in ready_replaced_functions:
                            if function_name in ready_functions:
                                fun_process = FunctionProcess(
                                    application_obj.placement[function_name],
                                    env,
                                    application_obj,
                                )
                                application_obj.function_processes[
                                    function_name
                                ] = fun_process

                        # application replaced - update infastructure.pl
                        logger.info("Application replaced - update infrastructure")
                        dump_infrastructure(
                            infrastructure, gc.SECF2F_INFRASTRUCTURE_PATH
                        )

                    else:
                        # set application state as canceled
                        application_obj.state = ApplicationState.CANCELED

                        # release resources of waiting and running functions which were not involved by the crash
                        for function_name in application_obj.placement:
                            function = application_obj.placement[function_name]
                            if function.state in [
                                FunctionState.WAITING,
                                FunctionState.RUNNING,
                            ]:
                                node_id = function.node_id
                                node_obj = application_obj.infrastructure_nodes[node_id]
                                node_obj.release_resources(
                                    function.memory, function.v_cpu
                                )

                                logger.info(
                                    "Application %s - Function %s has been canceled after failed replacement",
                                    application_obj.id,
                                    function.id,
                                )

                                # interrupt the function if it is running
                                if function.state == FunctionState.RUNNING:

                                    # set function as interrupted
                                    function.state = FunctionState.INTERRUPTED

                                    application_obj.function_processes[
                                        function.id
                                    ].action.interrupt()

                                else:
                                    # set function state as canceled
                                    function.state = FunctionState.CANCELED

        # PLACEMENT PHASE

        # for each event generator
        for event_generator in infrastructure.event_generators.values():

            # the event generator can't be triggered if it is inside a crashed node
            if event_generator.source_node in infrastructure.crashed_nodes:
                continue

            # generator is triggering?
            generator_triggered = take_decision(
                config.event_generator_trigger_probability
            )

            if generator_triggered:
                logger.info(f"Generator {event_generator.generator_id} triggered")

                # get the list of triggered events
                triggered_events = []
                for event, event_probability in event_generator.events:

                    # event of that generator is triggering?
                    event_triggered = take_decision(event_probability)

                    if event_triggered:
                        triggered_events.append(event)

                # for each application, check if it is triggered by one of these events
                for application_name in config.applications:

                    config_application = config.applications[application_name]

                    if config_application["trigger_event"] in triggered_events:

                        # try to place the application
                        logger.info(
                            f"Event {config_application['trigger_event']} ({event_generator.generator_id}) triggered {application_name}"
                        )

                        # get application path
                        application_filename = config_application["filename"]
                        application_path = os.path.join(
                            gc.APPLICATIONS_DIR, application_filename
                        )

                        # save application file into default path
                        shutil.copy(application_path, gc.SECF2F_APP_PATH)

                        # place
                        application_obj = place_application(
                            application_name,
                            config_application,
                            event_generator.generator_id,
                            infrastructure,
                            step_number,
                        )

                        if application_obj is not None:
                            # launch application

                            # get first functions to execute from the chain
                            ready_functions: list = get_ready_functions(
                                application_obj.chain
                            )

                            # set application as running
                            application_obj.state = ApplicationState.RUNNING

                            # execute ready functions
                            for function_name in ready_functions:
                                fun_process = FunctionProcess(
                                    application_obj.placement[function_name],
                                    env,
                                    application_obj,
                                )
                                application_obj.function_processes[
                                    function_name
                                ] = fun_process

                            # add application into the list
                            list_of_applications.append(application_obj)

                            # application placed - update infastructure.pl
                            logger.info("Application placed - update infrastructure")
                            dump_infrastructure(
                                infrastructure, gc.SECF2F_INFRASTRUCTURE_PATH
                            )

        # get nodes statistics
        for node in infrastructure.nodes.values():
            load = node.get_load()
            # in order to reduce the output report, don't save loads stat if load is 0
            if load > 0:
                node_data = {
                    "consumption": node.get_energy(),
                    "load": load,
                }
                node_stats[node.id][step_number] = node_data

        # make 1 step of simulation
        yield env.timeout(1)


def print_stats():
    """Print simulation stats into a report"""
    
    # get logger
    logger = logs.get_logger()

    logger.info("--- STATISTICS ---")

    # accumulators for all applications
    tempted_placements = {"placements": 0, "replacements": 0, "global": 0}
    successes = {"placements": 0, "replacements": 0, "global": 0}
    time_sum = {
        "placements": {"successes": 0, "failures": 0, "total": 0},
        "replacements": {"successes": 0, "failures": 0, "total": 0},
        "global": {"successes": 0, "failures": 0, "total": 0},
    }

    # calculate execution times

    applications_stats_keys = applications_stats.keys()

    for app_name in applications_stats_keys:

        application = applications_stats[app_name]

        # Scan application's placement and replacement data

        for placement_type in ["placements", "replacements"]:

            # application's data accumulators
            app_tempted_placements = len(application[placement_type]["data"])
            app_successes = 0
            app_time_sum = {"successes": 0, "failures": 0, "total": 0}

            placement_results = application[placement_type]["data"]

            for result in placement_results:

                execution_time = result["duration"]
                app_time_sum["total"] += execution_time

                if result["success"]:
                    app_time_sum["successes"] += execution_time
                    time_sum[placement_type]["successes"] += execution_time
                    app_successes += 1
                else:
                    app_time_sum["failures"] += execution_time
                    time_sum[placement_type]["failures"] += execution_time

            # enrich application stats
            application[placement_type]["attempts"] = app_tempted_placements
            application[placement_type]["successes"] = app_successes
            application[placement_type]["avg_execution_time"] = (
                0
                if app_tempted_placements == 0
                else float(f'{app_time_sum["total"]/app_tempted_placements}')
            )
            application[placement_type]["avg_success_execution_time"] = (
                0
                if app_successes == 0
                else float(f'{app_time_sum["successes"]/app_successes}')
            )
            application[placement_type]["avg_failure_execution_time"] = (
                0
                if app_successes == app_tempted_placements
                else float(
                    f'{app_time_sum["failures"]/(app_tempted_placements-app_successes)}'
                )
            )

            # accumulate variables
            tempted_placements["global"] += app_tempted_placements
            tempted_placements[placement_type] += app_tempted_placements

            successes["global"] += app_successes
            successes[placement_type] += app_successes

            time_sum[placement_type]["total"] += app_time_sum["total"]

            time_sum["global"]["successes"] += app_time_sum["successes"]
            time_sum["global"]["failures"] += app_time_sum["failures"]
            time_sum["global"]["total"] += app_time_sum["total"]

    # calculate average times
    avg_total_exec_time = {
        "placements": 0
        if tempted_placements["placements"] == 0
        else float(
            f"{time_sum['placements']['total']/tempted_placements['placements']}"
        ),
        "replacements": 0
        if tempted_placements["replacements"] == 0
        else float(
            f"{time_sum['replacements']['total']/tempted_placements['replacements']}"
        ),
        "global": 0
        if tempted_placements["global"] == 0
        else float(f"{time_sum['global']['total']/tempted_placements['global']}"),
    }
    avg_success_exec_time = {
        "placements": 0
        if successes["placements"] == 0
        else float(f"{time_sum['placements']['successes']/successes['placements']}"),
        "replacements": 0
        if successes["replacements"] == 0
        else float(
            f"{time_sum['replacements']['successes']/successes['replacements']}"
        ),
        "global": 0
        if successes["global"] == 0
        else float(f"{time_sum['global']['successes']/successes['global']}"),
    }
    avg_failure_exec_time = {
        "placements": 0
        if tempted_placements["placements"] == successes["placements"]
        else float(
            f"{time_sum['placements']['failures']/(tempted_placements['placements']-successes['placements'])}"
        ),
        "replacements": 0
        if tempted_placements["replacements"] == successes["replacements"]
        else float(
            f"{time_sum['replacements']['failures']/(tempted_placements['replacements']-successes['replacements'])}"
        ),
        "global": 0
        if tempted_placements["global"] == successes["global"]
        else float(
            f"{time_sum['global']['failures']/(tempted_placements['global']-successes['global'])}"
        ),
    }

    stats_to_dump = {
        "general": {
            "epochs": config.sim_num_of_epochs,
            "seed": config.sim_seed,
            "global": {
                "total_attempts": tempted_placements["global"],
                "total_successes": successes["global"],
                "avg_total_execution_time": avg_total_exec_time["global"],
                "avg_success_execution_time": avg_success_exec_time["global"],
                "avg_failure_execution_time": avg_failure_exec_time["global"],
            },
            "placements": {
                "total_attempts": tempted_placements["placements"],
                "total_successes": successes["placements"],
                "avg_total_execution_time": avg_total_exec_time["placements"],
                "avg_success_execution_time": avg_success_exec_time["placements"],
                "avg_failure_execution_time": avg_failure_exec_time["placements"],
            },
            "replacements": {
                "total_attempts": tempted_placements["replacements"],
                "total_successes": successes["replacements"],
                "avg_total_execution_time": avg_total_exec_time["replacements"],
                "avg_success_execution_time": avg_success_exec_time["replacements"],
                "avg_failure_execution_time": avg_failure_exec_time["replacements"],
            },
        },
        "nodes": {
            "load": node_stats,
            "events": node_events,
        },
        "links": {
            "events": link_events,
        },
        "applications": applications_stats,
    }

    logger.info("Total number of placements: %d" % tempted_placements["placements"])
    logger.info("Successess: %d" % successes["placements"])
    logger.info("Total number of replacements: %d" % tempted_placements["replacements"])
    logger.info("Successess: %d" % successes["replacements"])
    logger.info("Total number of both: %d" % tempted_placements["global"])
    logger.info("Successess: %d" % successes["global"])

    # write the report

    logger.info(
        f"Writing JSON's execution report on '{config.sim_report_output_file}'..."
    )

    with open(config.sim_report_output_file, "w") as file:
        json.dump(stats_to_dump, indent=4, default=str, fp=file)

def visualize_results():
    """Run the visualization script to create plots and analytics from simulation results"""
    logger = logs.get_logger()
    logger.info("Starting visualization of simulation results...")
    
    try:
        # Get the full path to the visualization script
        vis_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "scripts", "vis.py")
        
        # Check if the visualization script exists
        if not os.path.exists(vis_script_path):
            logger.error(f"Visualization script not found at {vis_script_path}")
            return
            
        # Execute the visualization script as a module
        import importlib.util
        spec = importlib.util.spec_from_file_location("vis", vis_script_path)
        vis_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vis_module)
        
        # Call the main function of the visualization script
        vis_module.main()
        
        logger.info("Visualization completed successfully")
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def export_csv_data():
    """Export simulation data to CSV files for analysis"""
    logger = logs.get_logger()
    logger.info("Exporting simulation data to CSV files...")
    
    # Create a directory for CSV exports if it doesn't exist
    export_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csv_exports")
    os.makedirs(export_dir, exist_ok=True)
    
    try:
        # Load the report data (using the same file generated by print_stats)
        with open(config.sim_report_output_file, 'r') as file:
            report_data = json.load(file)
        
        # 1. Export placement data
        placement_data = []
        replacement_data = []
        
        for app_name, app_data in report_data['applications'].items():
            # Extract placement data
            for placement in app_data['placements']['data']:
                record = {
                    'application': app_name,
                    'type': 'placement',
                    'node': placement['node'],
                    'epoch': placement['epoch'],
                    'duration': placement['duration'],
                    'success': placement['success']
                }
                placement_data.append(record)
            
            # Extract replacement data
            for replacement in app_data['replacements']['data']:
                record = {
                    'application': app_name,
                    'type': 'replacement',
                    'epoch': replacement['epoch'],
                    'duration': replacement['duration'],
                    'success': replacement['success'],
                    'crashed_nodes': ','.join(replacement['crashed_nodes']) if replacement['crashed_nodes'] else '',
                    'crashed_links': ','.join([f"{l[0]}-{l[1]}" for l in replacement['crashed_links']]) if replacement['crashed_links'] else ''
                }
                replacement_data.append(record)
        
        # Create DataFrames and export to CSV
        import pandas as pd
        
        if placement_data:
            placements_df = pd.DataFrame(placement_data)
            placements_csv_path = os.path.join(export_dir, 'placements.csv')
            placements_df.to_csv(placements_csv_path, index=False)
            logger.info(f"Exported placement data to {placements_csv_path}")
        
        if replacement_data:
            replacements_df = pd.DataFrame(replacement_data)
            replacements_csv_path = os.path.join(export_dir, 'replacements.csv')
            replacements_df.to_csv(replacements_csv_path, index=False)
            logger.info(f"Exported replacement data to {replacements_csv_path}")
        
        # 2. Export node data
        node_load_data = []
        
        for node_id, epochs in report_data['nodes']['load'].items():
            for epoch, stats in epochs.items():
                record = {
                    'node_id': node_id,
                    'epoch': int(epoch),
                    'load': stats['load'],
                    'consumption': stats.get('consumption', 0)
                }
                node_load_data.append(record)
        
        if node_load_data:
            node_load_df = pd.DataFrame(node_load_data)
            node_load_df = node_load_df.sort_values(['node_id', 'epoch'])
            node_load_csv_path = os.path.join(export_dir, 'node_load.csv')
            node_load_df.to_csv(node_load_csv_path, index=False)
            logger.info(f"Exported node load data to {node_load_csv_path}")
        
        # 3. Export node event data
        if report_data['nodes']['events']:
            node_events_df = pd.DataFrame(report_data['nodes']['events'])
            node_events_df = node_events_df.sort_values(['epoch', 'node_id'])
            node_events_csv_path = os.path.join(export_dir, 'node_events.csv')
            node_events_df.to_csv(node_events_csv_path, index=False)
            logger.info(f"Exported node events data to {node_events_csv_path}")
        
        # 4. Export link event data
        if report_data['links']['events']:
            link_events_df = pd.DataFrame(report_data['links']['events'])
            link_events_df = link_events_df.sort_values('epoch')
            link_events_csv_path = os.path.join(export_dir, 'link_events.csv')
            link_events_df.to_csv(link_events_csv_path, index=False)
            logger.info(f"Exported link events data to {link_events_csv_path}")
        
        # 5. Export application summary data
        app_summary_data = []
        
        for app_name, app_data in report_data['applications'].items():
            record = {
                'application': app_name,
                'placement_attempts': app_data['placements']['attempts'],
                'placement_successes': app_data['placements']['successes'],
                'placement_success_rate': app_data['placements']['successes'] / app_data['placements']['attempts'] if app_data['placements']['attempts'] > 0 else 0,
                'avg_placement_time': app_data['placements']['avg_execution_time'],
                'replacement_attempts': app_data['replacements']['attempts'],
                'replacement_successes': app_data['replacements']['successes'],
                'replacement_success_rate': app_data['replacements']['successes'] / app_data['replacements']['attempts'] if app_data['replacements']['attempts'] > 0 else 0,
                'avg_replacement_time': app_data['replacements']['avg_execution_time']
            }
            app_summary_data.append(record)
        
        if app_summary_data:
            app_summary_df = pd.DataFrame(app_summary_data)
            app_summary_csv_path = os.path.join(export_dir, 'application_summary.csv')
            app_summary_df.to_csv(app_summary_csv_path, index=False)
            logger.info(f"Exported application summary data to {app_summary_csv_path}")
        
        # 6. Export simulation overview data
        overview_data = {
            'epochs': [report_data['general']['epochs']],
            'seed': [report_data['general']['seed']],
            'total_placements': [report_data['general']['placements']['total_attempts']],
            'successful_placements': [report_data['general']['placements']['total_successes']],
            'placement_success_rate': [report_data['general']['placements']['total_successes'] / report_data['general']['placements']['total_attempts'] 
                                      if report_data['general']['placements']['total_attempts'] > 0 else 0],
            'total_replacements': [report_data['general']['replacements']['total_attempts']],
            'successful_replacements': [report_data['general']['replacements']['total_successes']],
            'replacement_success_rate': [report_data['general']['replacements']['total_successes'] / report_data['general']['replacements']['total_attempts']
                                        if report_data['general']['replacements']['total_attempts'] > 0 else 0],
            'avg_placement_time': [report_data['general']['placements']['avg_total_execution_time']],
            'avg_replacement_time': [report_data['general']['replacements']['avg_total_execution_time']],
        }
        
        overview_df = pd.DataFrame(overview_data)
        overview_csv_path = os.path.join(export_dir, 'simulation_overview.csv')
        overview_df.to_csv(overview_csv_path, index=False)
        logger.info(f"Exported simulation overview data to {overview_csv_path}")
        
        # Create a README file explaining the CSV contents
        readme_content = """# LambdaFogSim Exported CSV Data

This directory contains CSV exports of the LambdaFogSim simulation results:

1. `placements.csv` - Details of all application placement attempts
2. `replacements.csv` - Details of all application replacement attempts
3. `node_load.csv` - Node load statistics per epoch
4. `node_events.csv` - Node crash and resurrection events
5. `link_events.csv` - Link crash and resurrection events
6. `application_summary.csv` - Performance summary for each application
7. `simulation_overview.csv` - Overall simulation statistics

These files can be used for further analysis with tools like pandas, R, or Excel.
"""
        
        with open(os.path.join(export_dir, 'README.md'), 'w') as readme_file:
            readme_file.write(readme_content)
        
        logger.info(f"All CSV data exported successfully to {export_dir}/")
        
    except Exception as e:
        logger.error(f"Error exporting CSV data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
# Entry point of the simulator
def main():

    # INITIALIZATION PHASE - Global variables

    gc.init()

    # logging
    logger = logs.init_logger()

    # parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # if verbose mode is active show all logging messages
    if args.verbose:
        import logging

        logger.info("Verbose mode is now active, all messages will be shown")
        logger.setLevel(logging.INFO)

    # check if the given config path is a file
    if not os.path.exists(args.config) or not os.path.isfile(args.config):
        logger.error("Config path '%s' not exists or is not a file" % args.config)
        parser.print_help()
        return 1

    # parse the config file
    parsing_succeed = parse_config(args.config)

    if not parsing_succeed:
        logger.critical("Config parsing failed")
        return 1

    logger.info("Config correctly parsed")

    # Seed for deterministic execution
    random.seed(config.sim_seed)

    # Instance infrastructure

    infrastructure: Infrastructure

    if config.infr_type == "physical":
        logger.info("Infrastructure generation is starting")

        # check if the given physical infrastructure config path is a file
        if not os.path.exists(args.physical_infrastructure) or not os.path.isfile(
            args.physical_infrastructure
        ):
            logger.error(
                f"Physical infrastructure config path '{args.physical_infrastructure}' not exists or is not a file"
            )
            parser.print_help()
            return 1

        # randomly generate the infrastructure
        infrastructure = generate_infrastructure(args.physical_infrastructure)

        if not infrastructure:
            logger.critical("Infrastructure generation failed...")
            return 1

        # save infrastructure file into default path
        dump_infrastructure(infrastructure, gc.SECF2F_INFRASTRUCTURE_PATH)

        logger.info(
            f"Infrastructure generated, it will be saved into {gc.GENERATED_INFRASTRUCTURE_PATH}"
        )
        shutil.copy(gc.SECF2F_INFRASTRUCTURE_PATH, gc.GENERATED_INFRASTRUCTURE_PATH)

    else:

        if not args.logical_infrastructure:
            logger.critical(
                "If infrastructure type is 'logical', a valid infrastructure file must be given"
            )
            parser.print_help()
            return 1

        # check if the given logical infrastructure path is a file
        if not os.path.exists(args.logical_infrastructure) or not os.path.isfile(
            args.logical_infrastructure
        ):
            logger.error(
                f"Logical infrastructure path '{args.logical_infrastructure}' not exists or is not a file"
            )
            parser.print_help()
            return 1

        # check if the given logical infrastructure path has .pl
        if str(args.logical_infrastructure).split(".")[-1].lower() != "pl":
            logger.error(
                f"Logical infrastructure path '{args.logical_infrastructure}' is not a Prolog file"
            )
            parser.print_help()
            return 1

        # load infrastructure from the Prolog file
        infrastructure = LogicalInfrastructure.loads(args.logical_infrastructure)

        # save infrastructure file into default path
        shutil.copy(args.logical_infrastructure, gc.SECF2F_INFRASTRUCTURE_PATH)

    # initialize node stats dictionary
    for node_id in infrastructure.nodes:
        node_stats[node_id] = {}

    # SIMPY PHASE

    # Instance an enviroment
    env = simpy.Environment()

    # start simulation
    env.process(simulation(env, config.sim_num_of_epochs, infrastructure))

    # we simulate for sim_num_of_epochs epochs
    env.run(until=config.sim_num_of_epochs)

    # STATISTICS

    print_stats()

    export_csv_data()

    # visualize_results()
    
    logger.info("Simulation done! Byee :)")

    return 0


if __name__ == "__main__":
    main()
