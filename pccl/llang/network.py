# from typing import List, Dict, Tuple, Any, Optional, Set
# import itertools
# import threading
# from pccl.logging import get_logger

# logger = get_logger("heterdist.network")

# _thread_local = threading.local()
# _thread_local.current_topology = None
# _thread_local.current_route_policy = None

# class Device:
#     _ids = itertools.count()

#     def __init__(self, name: str, num_ports: int, device_type: str = "generic"):
#         self.id = next(Device._ids)
#         self.name = name
#         self.num_ports = num_ports
#         self.device_type = device_type

#     def __repr__(self):
#         return "%s(id=%d, name='%s', ports=%d)" % (
#             self.__class__.__name__, self.id, self.name, self.num_ports
#         )

# class Node(Device):
#     def __init__(self, name: str, num_ports: int, primary_ip: Optional[str] = None,
#                  secondary_ip: Optional[str] = None):
#         super().__init__(name, num_ports, device_type="node")
#         self.primary_ip = primary_ip
#         self.secondary_ip = secondary_ip

#     def set_ips(self, primary_ip: str, secondary_ip: Optional[str] = None):
#         self.primary_ip = primary_ip
#         self.secondary_ip = secondary_ip
#         return self

#     def __repr__(self):
#         return "Node(id=%d, name='%s', primary_ip=%s, secondary_ip=%s)" % (
#             self.id, self.name, self.primary_ip, self.secondary_ip
#         )

# class Switch(Device):
#     def __init__(self, name: str, num_ports: int, switch_type: str = "generic_switch"):
#         super().__init__(name, num_ports, device_type=switch_type)
#         self.flow_table: Dict[Tuple[int, Optional[str], Optional[str]], List[int]] = {}

#     def add_flow_rule(self, in_port: int, src_ip: Optional[str], dst_ip: Optional[str],
#                      out_ports: List[int]):
#         key = (in_port, src_ip, dst_ip)
#         self.flow_table[key] = out_ports
#         return self

#     def match_flow(self, in_port: int, src_ip: Optional[str], dst_ip: Optional[str]) -> List[int]:
#         if (in_port, src_ip, dst_ip) in self.flow_table:
#             return self.flow_table[(in_port, src_ip, dst_ip)]
#         if (in_port, None, dst_ip) in self.flow_table:
#             return self.flow_table[(in_port, None, dst_ip)]
#         if (in_port, src_ip, None) in self.flow_table:
#             return self.flow_table[(in_port, src_ip, None)]
#         if (in_port, None, None) in self.flow_table:
#             return self.flow_table[(in_port, None, None)]
#         return []

# class NonReconfigurableSwitch(Switch):
#     def __init__(self, name: str, num_ports: int):
#         super().__init__(name, num_ports, switch_type="non_reconfigurable_switch")

#     def get_internal_connections(self) -> Dict[int, Set[int]]:
#         connections = {}
#         for i in range(self.num_ports):
#             connections[i] = {j for j in range(self.num_ports) if j != i}
#         return connections

# class ReconfigurableSwitch(Switch):
#     def __init__(self, name: str, num_ports: int):
#         super().__init__(name, num_ports, switch_type="reconfigurable_switch")
#         self._current_config: Set[Tuple[int, int]] = set()
#         self._port_connections: Dict[int, Optional[int]] = {p: None for p in range(num_ports)}

#     def configure(self, config: Set[Tuple[int, int]]):
#         self._current_config = set()
#         self._port_connections = {p: None for p in range(self.num_ports)}
#         for p1, p2 in config:
#             if not (0 <= p1 < self.num_ports and 0 <= p2 < self.num_ports):
#                 logger.warning("Invalid port pair (%d, %d) for switch %d. Ignoring.",
#                              p1, p2, self.id)
#                 continue
#             if self._port_connections[p1] is not None:
#                 logger.warning("Port %d already connected to port %d. Ignoring new connection.",
#                              p1, self._port_connections[p1])
#                 continue
#             if self._port_connections[p2] is not None:
#                 logger.warning("Port %d already connected to port %d. Ignoring new connection.",
#                              p2, self._port_connections[p2])
#                 continue
#             if p1 == p2:
#                 logger.warning("Cannot connect port %d to itself. Ignoring.", p1)
#                 continue
#             pair = (min(p1, p2), max(p1, p2))
#             self._current_config.add(pair)
#             self._port_connections[p1] = p2
#             self._port_connections[p2] = p1

#     def get_internal_connections(self) -> Dict[int, Set[int]]:
#         connections: Dict[int, Set[int]] = {p: set() for p in range(self.num_ports)}
#         for port, connected_port in self._port_connections.items():
#             if connected_port is not None:
#                 connections[port].add(connected_port)
#         return connections

# class Cluster:
#     def __init__(self, name: str = "default_cluster"):
#         self.name = name
#         self.devices: Dict[int, Device] = {}
#         self.nodes: Dict[int, Node] = {}
#         self.switches: Dict[int, Switch] = {}
#         self.physical_connections: Dict[Tuple[int, int], Tuple[int, int, Dict[str, Any]]] = {}
#         self._topologies: Dict[str, 'Topology'] = {}

#     def add_node(self, node: Node):
#         if node.id in self.devices:
#             logger.warning("Device ID %d already exists. Overwriting.", node.id)
#         self.devices[node.id] = node
#         self.nodes[node.id] = node
#         return self

#     def add_switch(self, switch: Switch):
#         if switch.id in self.devices:
#             logger.warning("Device ID %d already exists. Overwriting.", switch.id)
#         self.devices[switch.id] = switch
#         self.switches[switch.id] = switch
#         return self

#     def add_physical_connection(self, dev1_id: int, port1_idx: int, dev2_id: int,
#                               port2_idx: int, **attrs):
#         if dev1_id not in self.devices or dev2_id not in self.devices:
#             logger.warning("Cannot connect non-existent devices: %d or %d", dev1_id, dev2_id)
#             return self

#         dev1 = self.devices[dev1_id]
#         dev2 = self.devices[dev2_id]

#         if not (0 <= port1_idx < dev1.num_ports and 0 <= port2_idx < dev2.num_ports):
#             logger.warning("Invalid port index for connection (%d,%d) <-> (%d,%d)",
#                          dev1_id, port1_idx, dev2_id, port2_idx)
#             return self

#         port1_key = (dev1_id, port1_idx)
#         port2_key = (dev2_id, port2_idx)

#         if port1_key in self.physical_connections or port2_key in self.physical_connections:
#             existing1 = self.physical_connections.get(port1_key)
#             existing2 = self.physical_connections.get(port2_key)
#             if not ((existing1 and existing1[:2] == (dev2_id, port2_idx)) or \
#                     (existing2 and existing2[:2] == (dev1_id, port1_idx))):
#                 logger.warning("Port %s or %s already connected. Cannot change physical connections.",
#                              port1_key, port2_key)
#                 return self
#         self.physical_connections[port1_key] = (dev2_id, port2_idx, attrs)
#         self.physical_connections[port2_key] = (dev1_id, port1_idx, attrs.copy())
#         return self

#     def topology(self, label: str) -> 'Topology':
#         if label not in self._topologies:
#             self._topologies[label] = Topology(label, self)
#         return self._topologies[label]

# class Topology:
#     def __init__(self, label: str, cluster: Optional[Cluster] = None):
#         self.label = label
#         self.cluster = cluster
#         self.reconfig_switch_paths = {}  # 只存储可重配置交换机的内部通路
#         self._route_policies: Dict[str, 'RoutePolicy'] = {}
#         self._previous_topology = None
#         logger.info("Creating topology: %s", label)

#     def add_reconfig_switch_path(self, switch_id: int, paths: Set[Tuple[int, int]]):
#         self.reconfig_switch_paths[switch_id] = paths
#         logger.info("Adding reconfigurable switch path for switch %d: %d paths",
#                    switch_id, len(paths))
#         return self

#     def get_reconfig_switch_paths(self) -> Dict[int, Set[Tuple[int, int]]]:
#         return self.reconfig_switch_paths

#     def route_policy(self, label: str, **kwargs) -> 'RoutePolicy':
#         if label not in self._route_policies:
#             self._route_policies[label] = RoutePolicy(label, self, **kwargs)
#         return self._route_policies[label]

#     def serialize(self) -> Dict[str, Any]:
#         return {
#             "label": self.label,
#             "reconfig_switch_paths": {k: list(v) for k, v in self.reconfig_switch_paths.items()}
#         }

#     @classmethod
#     def deserialize(cls, data: Dict[str, Any], cluster: Optional[Cluster] = None):
#         topo = cls(data["label"], cluster)
#         topo.reconfig_switch_paths = {k: set(v) for k, v in data.get("reconfig_switch_paths", {}).items()}
#         return topo

#     def __enter__(self):
#         self._previous_topology = getattr(_thread_local, 'current_topology', None)
#         _thread_local.current_topology = self
#         logger.debug("Entering topology context: %s", self.label)
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         _thread_local.current_topology = self._previous_topology
#         logger.debug("Exiting topology context: %s", self.label)
#         return False

# class RoutePolicy:
#     def __init__(self, algorithm: str, topology: Optional[Topology] = None, **kwargs):
#         self.algorithm = algorithm
#         self.topology = topology
#         self.params = kwargs
#         self.flow_tables = {}
#         self._previous_policy = None
#         logger.info("Creating route policy: %s", algorithm)

#     def set_flow_tables(self, flow_tables: Dict[int, List[Dict[str, Any]]]):
#         self.flow_tables = flow_tables
#         logger.info("Setting flow tables: %d switches", len(flow_tables))
#         return self

#     def get_flow_tables(self) -> Dict[int, List[Dict[str, Any]]]:
#         return self.flow_tables

#     def add_flow_rule(self, switch_id: int, match: Dict[str, Any], action: Dict[str, Any]):
#         if switch_id not in self.flow_tables:
#             self.flow_tables[switch_id] = []
#         self.flow_tables[switch_id].append({"match": match, "action": action})
#         return self

#     def clear_flow_tables(self):
#         self.flow_tables = {}
#         return self

#     def serialize(self) -> Dict[str, Any]:
#         return {
#             "algorithm": self.algorithm,
#             "params": self.params.copy(),
#             "flow_tables": self.flow_tables.copy()
#         }

#     @classmethod
#     def deserialize(cls, data: Dict[str, Any], topology: Optional[Topology] = None):
#         policy = cls(data["algorithm"], topology, **data["params"])
#         policy.flow_tables = data["flow_tables"]
#         return policy

#     def __enter__(self):
#         self._previous_policy = getattr(_thread_local, 'current_route_policy', None)
#         _thread_local.current_route_policy = self
#         logger.debug("Entering route policy context: %s", self.algorithm)
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         _thread_local.current_route_policy = self._previous_policy
#         logger.debug("Exiting route policy context: %s", self.algorithm)
#         return False

# def get_current_topology() -> Optional[Topology]:
#     return getattr(_thread_local, 'current_topology', None)

# def get_current_route_policy() -> Optional[RoutePolicy]:
#     return getattr(_thread_local, 'current_route_policy', None)
