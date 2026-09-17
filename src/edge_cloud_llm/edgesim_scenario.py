"""Step 8: run the same middleware inside an EdgeSimPy scenario.

EdgeSimPy (section 9's mandated evaluation tool) has no built-in notion of
token-by-token LLM generation, so the integration point kept here is
deliberately thin (section 5 - no speculative infrastructure): a minimal
two-node topology (edge server <-> cloud server) with a network link carrying
bandwidth/delay, and a resource-management callback that runs one full
middleware.generate() call per simulated tick.
"""

from __future__ import annotations

from dataclasses import dataclass

import edge_sim_py as esp

from .middleware import GenerationResult, MiddlewareCore


@dataclass
class NetworkScenario:
    bandwidth_mbps: float
    delay_ms: float


def _reset_edgesimpy_registries() -> None:
    """EdgeSimPy components register themselves in class-level lists; reset
    them so repeated scenario runs (e.g. across tests) start from a clean
    slate, matching what Simulator.initialize() itself does."""
    for component_class in esp.component_manager.ComponentManager.__subclasses__():
        if component_class.__name__ != "Simulator":
            component_class._object_count = 0
            component_class._instances = []


def run_scenario(
    middleware: MiddlewareCore,
    prompt: str,
    scenario: NetworkScenario,
    num_ticks: int = 2,
    logs_directory: str = "logs",
) -> list[GenerationResult]:
    """Runs `num_ticks` simulated ticks; each tick generates one full
    response through the middleware under the given network conditions."""
    _reset_edgesimpy_registries()

    edge_node = esp.EdgeServer(model_name="edge-server")
    cloud_node = esp.EdgeServer(model_name="cloud-server")

    link = esp.NetworkLink()
    link["nodes"] = [edge_node, cloud_node]
    link["bandwidth"] = scenario.bandwidth_mbps
    link["delay"] = scenario.delay_ms

    topology = esp.Topology()
    topology.add_node(edge_node)
    topology.add_node(cloud_node)
    topology.add_edge(edge_node, cloud_node)
    topology._adj[edge_node][cloud_node] = link
    topology._adj[cloud_node][edge_node] = link

    generations: list[GenerationResult] = []

    def resource_management_algorithm(parameters: dict) -> None:
        generations.append(middleware.generate(prompt))

    simulator = esp.Simulator(
        resource_management_algorithm=resource_management_algorithm,
        stopping_criterion=lambda sim: sim.schedule.steps >= num_ticks,
        tick_duration=1,
        logs_directory=logs_directory,
    )
    simulator.topology = topology
    for node in (edge_node, cloud_node):
        simulator.initialize_agent(agent=node)
    simulator.initialize_agent(agent=link)
    simulator.initialize_agent(agent=topology)

    simulator.run_model()
    return generations
