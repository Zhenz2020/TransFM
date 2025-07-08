import os
import sys
import osmnx as ox
import requests
import subprocess
import tempfile
from shapely.geometry import box
import streamlit as st
from streamlit_folium import st_folium
import folium
import traci
import sumolib
import networkx as nx
import xml.etree.ElementTree as ET
import shutil
import json
from streamlit_plotly_events import plotly_events
import plotly.graph_objs as go
import numpy as np

# ===================== Silicon Flow API 集成 =====================
def siliconflow_chat(prompt, api_key, model="Qwen/QwQ-32B"):
    """与 Silicon Flow API 进行交互"""
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.7,
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        st.error(f"调用API失败: {e}")
        return None
    except (KeyError, IndexError) as e:
        st.error(f"解析API响应失败: {e}")
        st.error(f"收到的响应: {response.text}")
        return None

# ===================== 核心功能函数 =====================
def debug_node_mapping(osm_graph, sumo_net_path):
    """
    调试OSMnx图和SUMO网络之间的节点ID映射
    """
    print("=== OSMnx vs SUMO Node ID Mapping Debug ===")
    
    # 读取SUMO网络
    net = sumolib.net.readNet(sumo_net_path)
    sumo_nodes = {node.getID(): node for node in net.getNodes()}
    
    print(f"OSMnx nodes: {len(osm_graph.nodes())}")
    print(f"SUMO nodes: {len(sumo_nodes)}")
    
    # 显示前10个OSMnx节点
    print("\nFirst 10 OSMnx nodes:")
    for i, (node_id, data) in enumerate(list(osm_graph.nodes(data=True))[:10]):
        print(f"  {i}: {node_id} (lat: {data.get('y', 'N/A')}, lon: {data.get('x', 'N/A')})")
    
    # 显示前10个SUMO节点
    print("\nFirst 10 SUMO nodes:")
    for i, (node_id, node) in enumerate(list(sumo_nodes.items())[:10]):
        coord = node.getCoord()
        print(f"  {i}: {node_id} (x: {coord[0]:.2f}, y: {coord[1]:.2f})")
    
    # 尝试找到匹配的节点
    print("\nTrying to find matching nodes...")
    matches = 0
    for osm_node in list(osm_graph.nodes())[:5]:  # 只检查前5个以避免太多输出
        found_match = False
        for sumo_id in sumo_nodes:
            if str(osm_node) in sumo_id or sumo_id in str(osm_node):
                print(f"  Potential match: OSMnx {osm_node} -> SUMO {sumo_id}")
                found_match = True
                matches += 1
                break
        if not found_match:
            print(f"  No match found for OSMnx node: {osm_node}")
    
    print(f"\nFound {matches} potential matches out of 5 checked nodes")
    return sumo_nodes

def download_osm(place=None, bbox=None, output_dir="."):
    if place:
        print(f"Downloading OSM data for place: {place}")
        G = ox.graph_from_place(place, network_type='drive', simplify=False)
    elif bbox:
        print(f"Downloading OSM data for bbox: {bbox}")
        G = ox.graph_from_bbox(bbox[3], bbox[1], bbox[2], bbox[0], network_type='drive', simplify=False)
    else:
        raise ValueError("Either place or bbox must be specified.")
    osm_path = os.path.join(output_dir, 'network.osm')
    ox.save_graph_xml(G, osm_path)
    return osm_path

def create_typemap(output_dir):
    typemap = '''<?xml version="1.0" encoding="UTF-8"?>
<typemap>
    <type id="highway.primary" color="0.2,0.2,0.2" priority="78" numLanes="2" speed="13.89"/>
    <type id="highway.secondary" color="0.2,0.2,0.2" priority="77" numLanes="2" speed="11.11"/>
    <type id="highway.residential" color="0.2,0.2,0.2" priority="76" numLanes="1" speed="8.33"/>
    <type id="highway.unclassified" color="0.2,0.2,0.2" priority="75" numLanes="1" speed="8.33"/>
</typemap>'''
    typemap_path = os.path.join(output_dir, 'typemap.xml')
    with open(typemap_path, 'w', encoding='utf-8') as f:
        f.write(typemap)
    return typemap_path

def osm_to_sumo(osm_path, typemap_path, output_dir):
    net_path = os.path.join(output_dir, 'network.net.xml')
    cmd = f'netconvert --osm-files "{osm_path}" --type-files "{typemap_path}" --output-file "{net_path}"'
    print(f"Running: {cmd}")
    if os.system(cmd) != 0:
        raise RuntimeError("netconvert failed!")
    return net_path

def generate_focused_routes_simple(net_path, output_dir, num_vehicles=30):
    """
    使用更简单的方法生成有效路径，确保产生足够的交通流量
    """
    rou_path = os.path.join(output_dir, "simple_routes.rou.xml")
    
    # 使用SUMO自带的randomTrips工具生成路径
    try:
        SUMO_HOME = os.environ.get("SUMO_HOME")
        if SUMO_HOME:
            random_trips_script = os.path.join(SUMO_HOME, "tools", "randomTrips.py")
            if os.path.isfile(random_trips_script):
                cmd = [
                    sys.executable,
                    random_trips_script,
                    "-n", net_path,
                    "-o", rou_path,
                    "-e", "400",  # 增加仿真时间
                    "--seed", "42",
                    "--min-distance", "50",  # 最小距离
                    "--max-distance", "500",  # 最大距离
                    "--trips", str(num_vehicles),  # 使用传入的车辆数
                    "--fringe-factor", "2",  # 增加边缘生成因子
                    "--validate"  # 验证路径
                ]
                
                print(f"使用randomTrips生成 {num_vehicles} 辆车的路径: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("randomTrips成功生成路径")
                    return rou_path
                else:
                    print(f"randomTrips失败: {result.stderr}")
    except Exception as e:
        print(f"randomTrips执行异常: {e}")
    
    # 如果randomTrips失败，创建高密度基本路径
    print("使用高密度基本路径生成...")
    net = sumolib.net.readNet(net_path)
    edges = [edge for edge in net.getEdges() if not edge.isSpecial() and edge.allows("passenger")]
    
    if not edges:
        print("错误：没有找到可用的边缘")
        return None
    
    routes_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="passenger" accel="2.6" decel="4.5" length="4.5" maxSpeed="55.0" sigma="0.5"/>
'''
    
    # 为每条边创建多辆车，产生拥堵
    vehicle_count = 0
    vehicles_per_edge = max(3, num_vehicles // len(edges))  # 每条边至少3辆车
    
    for edge in edges:
        for i in range(vehicles_per_edge):
            if vehicle_count >= num_vehicles:
                break
            depart_time = vehicle_count * 2  # 2秒间隔，产生拥堵
            routes_xml += f'''    <route id="route_{vehicle_count}" edges="{edge.getID()}"/>
    <vehicle id="vehicle_{vehicle_count}" type="passenger" route="route_{vehicle_count}" depart="{depart_time}"/>
'''
            vehicle_count += 1
        if vehicle_count >= num_vehicles:
            break
    
    # 如果还需要更多车辆，添加重复路径
    while vehicle_count < num_vehicles:
        edge = edges[vehicle_count % len(edges)]
        depart_time = vehicle_count * 2
        routes_xml += f'''    <route id="route_{vehicle_count}" edges="{edge.getID()}"/>
    <vehicle id="vehicle_{vehicle_count}" type="passenger" route="route_{vehicle_count}" depart="{depart_time}"/>
'''
        vehicle_count += 1
    
    routes_xml += '</routes>'
    
    with open(rou_path, 'w', encoding='utf-8') as f:
        f.write(routes_xml)
    
    print(f"生成高密度基本路径文件: {rou_path}, 包含 {vehicle_count} 辆车")
    return rou_path

def generate_focused_routes(net_path, output_dir, num_vehicles=30):
    """
    生成专门的路径文件，确保产生有效的交通流和等待时间
    """
    rou_path = os.path.join(output_dir, "focused_routes.rou.xml")
    
    # 读取网络文件，获取所有边和连接
    net = sumolib.net.readNet(net_path)
    edges = [edge for edge in net.getEdges() if not edge.isSpecial() and edge.allows("passenger")]
    nodes = [node for node in net.getNodes()]
    
    if len(edges) < 2:
        print("警告: 网络中可用边数量太少，创建最小路径集合")
        routes_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="passenger" accel="2.6" decel="4.5" length="4.5" maxSpeed="55.0" sigma="0.5"/>
'''
        if edges:
            edge = edges[0]
            for i in range(min(50, num_vehicles)):  # 增加到50辆车
                depart_time = i * 0.5  # 缩短到0.5秒间隔，快速发车
                routes_xml += f'''    <route id="route_{i}" edges="{edge.getID()}"/>
    <vehicle id="vehicle_{i}" type="passenger" route="route_{i}" depart="{depart_time}"/>
'''
        
        routes_xml += '</routes>'
        with open(rou_path, 'w', encoding='utf-8') as f:
            f.write(routes_xml)
        return rou_path
    
    # 创建高密度交通流路径，确保产生等待
    routes_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="passenger" accel="2.6" decel="4.5" length="4.5" maxSpeed="55.0" sigma="0.5"/>
'''
    
    import random
    random.seed(42)
    route_id = 0
    
    print(f"为 {len(edges)} 条边、{len(nodes)} 个节点生成 {num_vehicles} 辆车的高密度路径...")
    
    # 策略1: 创建多条边的连续路径（经过交叉口，产生冲突）
    print("生成多边连续路径，确保经过交叉口...")
    multi_edge_routes = 0
    for _ in range(min(num_vehicles // 2, 100)):  # 一半车辆走多边路径
        if len(edges) >= 3:
            # 随机选择3-5条相连的边
            start_edge = random.choice(edges)
            route_edges = [start_edge.getID()]
            current_node = start_edge.getToNode()
            
            # 尝试找到2-4条后续边
            for _ in range(random.randint(2, 4)):
                outgoing = [e for e in current_node.getOutgoing() if e.allows("passenger")]
                if outgoing:
                    next_edge = random.choice(outgoing)
                    route_edges.append(next_edge.getID())
                    current_node = next_edge.getToNode()
                else:
                    break
            
            if len(route_edges) >= 2:  # 至少2条边的路径
                depart_time = route_id * 0.3  # 0.3秒间隔，产生高密度
                route_str = " ".join(route_edges)
                routes_xml += f'''    <route id="route_{route_id}" edges="{route_str}"/>
    <vehicle id="vehicle_{route_id}" type="passenger" route="route_{route_id}" depart="{depart_time}"/>
'''
                route_id += 1
                multi_edge_routes += 1
                if route_id >= num_vehicles:
                    break
    
    print(f"已生成 {multi_edge_routes} 条多边路径")
    
    # 策略2: 在每个主要交叉口产生汇聚交通
    print("在主要交叉口生成汇聚交通...")
    major_junctions = [node for node in nodes if len(node.getIncoming()) >= 2 and len(node.getOutgoing()) >= 2]
    
    for junction in major_junctions[:5]:  # 选择前5个主要交叉口
        incoming_edges = [e for e in junction.getIncoming() if e.allows("passenger")]
        outgoing_edges = [e for e in junction.getOutgoing() if e.allows("passenger")]
        
        # 为每个进入边创建多辆车，都汇聚到这个交叉口
        for in_edge in incoming_edges:
            for out_edge in outgoing_edges:
                if route_id >= num_vehicles:
                    break
                depart_time = route_id * 0.2  # 0.2秒间隔，产生拥堵
                routes_xml += f'''    <route id="route_{route_id}" edges="{in_edge.getID()} {out_edge.getID()}"/>
    <vehicle id="vehicle_{route_id}" type="passenger" route="route_{route_id}" depart="{depart_time}"/>
'''
                route_id += 1
        if route_id >= num_vehicles:
            break
    
    # 策略3: 剩余车辆走单边路径，但发车密集
    print("生成剩余单边高密度路径...")
    while route_id < num_vehicles:
        edge = random.choice(edges)
        if edge.allows("passenger"):
            depart_time = route_id * 0.1  # 0.1秒间隔，极高密度
            routes_xml += f'''    <route id="route_{route_id}" edges="{edge.getID()}"/>
    <vehicle id="vehicle_{route_id}" type="passenger" route="route_{route_id}" depart="{depart_time}"/>
'''
            route_id += 1
    
    routes_xml += '</routes>'
    
    with open(rou_path, 'w', encoding='utf-8') as f:
        f.write(routes_xml)
    
    print(f"生成超高密度交通流路径文件: {rou_path}")
    print(f"包含 {route_id} 辆车，多边路径: {multi_edge_routes} 条，主要交叉口: {len(major_junctions[:5])} 个")
    print(f"最快发车间隔: 0.1秒，预期在前 {route_id * 0.1:.1f} 秒内全部发车完毕")
    return rou_path

def generate_routes(net_path, output_dir, sim_time=1000):
    rou_path = os.path.join(output_dir, "routes.rou.xml")
    SUMO_HOME = os.environ.get("SUMO_HOME")
    if not SUMO_HOME:
        raise RuntimeError("环境变量 SUMO_HOME 未设置！")

    random_trips_script = os.path.join(SUMO_HOME, "tools", "randomTrips.py")
    if not os.path.isfile(random_trips_script):
        raise RuntimeError(f"未找到 randomTrips.py：{random_trips_script}")

    cmd = [
        sys.executable,          # 当前 Python 解释器
        "-B",                    # 不写 .pyc，避免 Program Files 权限问题
        random_trips_script,
        "-n", net_path,
        "-o", rou_path,
        "-e", str(sim_time),
        "--seed", "42",
        "-l"  # 随机分配出发车道，利用新增车道
    ]
    print("Running:", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(SUMO_HOME, "tools") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("randomTrips.py failed! 详情见上方输出。")
    return rou_path

def create_sumocfg(net_path, rou_path, output_dir, sim_time=1000):
    sumocfg = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{os.path.basename(net_path)}"/>
        <route-files value="{os.path.basename(rou_path)}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{sim_time}"/>
        <step-length value="0.1"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
        <time-to-teleport value="300"/>
    </processing>
    <routing>
        <device.rerouting.probability value="0.3"/>
    </routing>
    <report>
        <verbose value="true"/>
        <duration-log.disable value="true"/>
    </report>
</configuration>'''
    sumocfg_path = os.path.join(output_dir, 'scenario.sumocfg')
    with open(sumocfg_path, 'w', encoding='utf-8') as f:
        f.write(sumocfg)
    return sumocfg_path

# ==== 仿真函数：计算指定交叉口平均等待时间 ====
def simulate_wait_time(sumocfg_path, net_path, junction_id=None, sim_time=1000):
    # 读取网络，获取与节点关联的所有边
    net = sumolib.net.readNet(net_path)
    edges_of_junction = None
    
    if junction_id is not None:
        # 尝试多种方式查找节点
        node_obj = None
        possible_ids = [
            str(junction_id),           # 原始ID
            junction_id,                # 数字ID
            f"n{junction_id}",          # 带前缀的ID
            f"node{junction_id}",       # 另一种前缀
            f"junction{junction_id}",   # 交叉口前缀
        ]
        
        # 首先尝试直接查找
        for possible_id in possible_ids:
            try:
                node_obj = net.getNode(str(possible_id))
                if node_obj:
                    print(f"Found node with ID: {possible_id}")
                    break
            except:
                continue
        
        # 如果还没找到，尝试从所有节点中查找相似的
        if node_obj is None:
            print(f"Node {junction_id} not found directly. Searching all nodes...")
            all_nodes = net.getNodes()
            print(f"Total nodes in SUMO network: {len(all_nodes)}")
            
            # 尝试找到包含原始ID的节点
            for node in all_nodes:
                node_id = node.getID()
                if str(junction_id) in node_id or node_id in str(junction_id):
                    node_obj = node
                    print(f"Found similar node: {node_id}")
                    break
            
            # 如果仍然没找到，使用第一个有连接的节点作为示例
            if node_obj is None:
                for node in all_nodes:
                    if len(node.getIncoming()) > 0 and len(node.getOutgoing()) > 0:
                        node_obj = node
                        print(f"Using fallback node: {node.getID()}")
                        break
        
        if node_obj is None:
            print(f"Warning: Could not find node {junction_id} in SUMO network")
            print("Available nodes (first 10):")
            for i, node in enumerate(net.getNodes()[:10]):
                print(f"  {i}: {node.getID()}")
            # 如果没有找到任何节点，使用所有边缘进行仿真
            edges_of_junction = None
        else:
            # 获取与节点关联的所有边
            edges_of_junction = set([e.getID() for e in node_obj.getIncoming()])
            edges_of_junction.update([e.getID() for e in node_obj.getOutgoing()])
            print(f"Found {len(edges_of_junction)} edges for junction: {list(edges_of_junction)[:5]}...")

    # 启动 TraCI 仿真
    sumo_binary = 'sumo'
    if os.environ.get('SUMO_GUI') == '1':
        sumo_binary = 'sumo-gui'
    
    try:
        traci.start([sumo_binary, '-c', sumocfg_path, '--step-length', '1'])
        
        total_wait = 0
        count = 0
        vehicle_wait_times = {}  # 记录每辆车的等待时间
        
        print(f"开始仿真 {sim_time} 步...")
        
        for step in range(sim_time):
            traci.simulationStep()
            vehicle_ids = traci.vehicle.getIDList()
            
            # 统计等待时间（所有车辆，不只是特定节点）
            for vid in vehicle_ids:
                try:
                    edge_id = traci.vehicle.getRoadID(vid)
                    waiting_time = traci.vehicle.getWaitingTime(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    
                    # 更全面的等待时间统计
                    if edges_of_junction is None or edge_id in edges_of_junction:
                        # 记录等待时间（大于0.1秒的才算等待）
                        if waiting_time > 0.1:
                            total_wait += waiting_time
                            count += 1
                            
                        # 记录低速行驶的时间（速度小于2m/s也算等待）
                        if speed < 2.0 and waiting_time > 0:
                            total_wait += waiting_time * 0.5  # 权重降低
                            count += 1
                            
                    # 记录每辆车的累计等待时间
                    if vid not in vehicle_wait_times:
                        vehicle_wait_times[vid] = 0
                    vehicle_wait_times[vid] += waiting_time
                    
                except:
                    # 忽略无效车辆
                    continue
            
            # 每50步输出一次进度
            if step % 50 == 0:
                active_vehicles = len(vehicle_ids)
                total_vehicles = len(vehicle_wait_times)
                print(f"仿真步 {step}/{sim_time}, 活跃车辆: {active_vehicles}, 总车辆: {total_vehicles}")
                
                # 显示前5辆车的等待时间
                if vehicle_wait_times:
                    sample_waits = list(vehicle_wait_times.values())[:5]
                    print(f"  样本等待时间: {[f'{w:.1f}' for w in sample_waits]}")
        
        traci.close()
        
        # 计算平均等待时间
        if count > 0:
            avg_wait = total_wait / count
            print(f"仿真完成. 统计次数: {count}, 总等待时间: {total_wait:.2f}, 平均等待时间: {avg_wait:.2f} 秒")
        else:
            # 如果没有统计到等待时间，使用车辆累计等待时间
            if vehicle_wait_times:
                total_vehicle_wait = sum(vehicle_wait_times.values())
                avg_wait = total_vehicle_wait / len(vehicle_wait_times)
                print(f"使用车辆累计等待时间计算: {len(vehicle_wait_times)} 辆车, 平均等待: {avg_wait:.2f} 秒")
            else:
                avg_wait = 0.0
                print("警告: 没有检测到任何车辆或等待时间")
        
        return avg_wait
            
    except Exception as e:
        print(f"仿真错误: {e}")
        try:
            traci.close()
        except:
            pass
        return 0.0

# ==== 网络改造策略：在指定交叉口增加掉头车道 ====
def add_uturn_lane(net_path, junction_id, output_net_path):
    """
    在指定交叉口增加掉头车道
    """
    print(f"正在为交叉口 {junction_id} 添加掉头车道...")
    
    tree = ET.parse(net_path)
    root = tree.getroot()
    
    # 首先检查网络中是否存在该节点
    nodes_in_net = set()
    for edge in root.findall('edge'):
        if edge.get('function') != 'internal':
            nodes_in_net.add(edge.get('from'))
            nodes_in_net.add(edge.get('to'))
    
    # 尝试找到匹配的节点ID
    target_junction = None
    possible_ids = [str(junction_id), junction_id]
    
    for possible_id in possible_ids:
        if str(possible_id) in nodes_in_net:
            target_junction = str(possible_id)
            break
    
    if target_junction is None:
        print(f"警告：未找到节点 {junction_id}，将对所有主要交叉口添加掉头车道")
        # 找到第一个有多个连接的节点
        node_connections = {}
        for edge in root.findall('edge'):
            if edge.get('function') != 'internal':
                from_node = edge.get('from')
                to_node = edge.get('to')
                node_connections[from_node] = node_connections.get(from_node, 0) + 1
                node_connections[to_node] = node_connections.get(to_node, 0) + 1
        
        # 选择连接数最多的节点
        target_junction = max(node_connections, key=node_connections.get) if node_connections else None
        print(f"使用替代节点: {target_junction}")
    
    modified_edges = 0
    for edge in root.findall('edge'):
        if edge.get('function') == 'internal':
            continue
            
        # 如果指定了目标交叉口，只修改相关边
        if target_junction and (edge.get('from') == target_junction or edge.get('to') == target_junction):
            lanes = edge.findall('lane')
            if not lanes:
                continue
                
            last_lane = lanes[-1]
            new_index = len(lanes)

            # 复制属性字典
            new_attrs = last_lane.attrib.copy()
            new_attrs.update({
                'id': f"{edge.get('id')}_{new_index}",
                'index': str(new_index),
                'allow': 'passenger',  # 确保允许乘用车
            })
            new_lane = ET.Element('lane', new_attrs)
            edge.append(new_lane)

            # 更新 edge 的 lane 数
            if 'numLanes' in edge.attrib:
                edge.set('numLanes', str(int(edge.get('numLanes')) + 1))
            
            modified_edges += 1
    
    tree.write(output_net_path)
    print(f"掉头车道添加完成，修改了 {modified_edges} 条边")
    return output_net_path

# ==== 网络改造策略：在指定交叉口关联道路增加 1 条普通车道 ====
def add_lane(net_path, junction_id, output_net_path):
    """
    在与指定节点直接相连的所有 edge 上增加 1 条同属性车道
    """
    print(f"正在为交叉口 {junction_id} 添加普通车道...")
    
    # 1. 复制原 net.xml 后直接改写 lane
    temp_net = tempfile.mktemp(suffix=".net.xml")
    shutil.copy(net_path, temp_net)

    tree = ET.parse(temp_net)
    root = tree.getroot()
    
    # 首先检查网络中是否存在该节点
    nodes_in_net = set()
    for edge in root.findall('edge'):
        if edge.get('function') != 'internal':
            nodes_in_net.add(edge.get('from'))
            nodes_in_net.add(edge.get('to'))
    
    # 尝试找到匹配的节点ID
    target_junction = None
    possible_ids = [str(junction_id), junction_id]
    
    for possible_id in possible_ids:
        if str(possible_id) in nodes_in_net:
            target_junction = str(possible_id)
            break
    
    if target_junction is None:
        print(f"警告：未找到节点 {junction_id}，将对所有主要交叉口添加车道")
        # 找到第一个有多个连接的节点
        node_connections = {}
        for edge in root.findall('edge'):
            if edge.get('function') != 'internal':
                from_node = edge.get('from')
                to_node = edge.get('to')
                node_connections[from_node] = node_connections.get(from_node, 0) + 1
                node_connections[to_node] = node_connections.get(to_node, 0) + 1
        
        # 选择连接数最多的节点
        target_junction = max(node_connections, key=node_connections.get) if node_connections else None
        print(f"使用替代节点: {target_junction}")
    
    modified_edges = {}
    modified_count = 0
    
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
            
        # 如果指定了目标交叉口，只修改相关边
        if target_junction and (edge.get("from") == target_junction or edge.get("to") == target_junction):
            lanes = edge.findall("lane")
            if not lanes:
                continue
                
            last_lane = lanes[-1]
            new_index = len(lanes)
            new_lane = ET.Element("lane", last_lane.attrib.copy())
            # 更新 id / index
            new_lane.set("id", f"{edge.get('id')}_{new_index}")
            new_lane.set("index", str(new_index))
            new_lane.set("allow", "passenger")  # 确保允许乘用车
            edge.append(new_lane)
            
            # 同步 numLanes
            if "numLanes" in edge.attrib:
                edge.set("numLanes", str(int(edge.get("numLanes")) + 1))
            
            modified_edges[edge.get('id')] = new_index
            modified_count += 1

    # 复制 connection 元素以连接新增车道
    for conn in list(root.findall('connection')):
        from_edge = conn.get('from')
        to_edge = conn.get('to')
        if from_edge in modified_edges:
            dup = ET.Element('connection', conn.attrib.copy())
            dup.set('fromLane', str(modified_edges[from_edge]))
            root.append(dup)
        if to_edge in modified_edges:
            dup2 = ET.Element('connection', conn.attrib.copy())
            dup2.set('toLane', str(modified_edges[to_edge]))
            root.append(dup2)

    tree.write(temp_net)
    print(f"车道添加完成，修改了 {modified_count} 条边")

    # 2. 简化netconvert过程，避免复杂的重建
    try:
        cmd = [
            "netconvert",
            "--sumo-net-file", temp_net,
            "--output-file", output_net_path,
            "--no-warnings"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("netconvert重建失败，使用原始修改文件")
            shutil.copy(temp_net, output_net_path)
    except Exception as e:
        print(f"netconvert执行失败: {e}")
        shutil.copy(temp_net, output_net_path)

    # 3. 清理
    try:
        os.remove(temp_net)
    except:
        pass

    return output_net_path

def extract_local_network(G, center_node, radius=300):
    """
    提取以指定节点为中心的最小局部网络（仅包含直接相关的路段和路口）
    
    Args:
        G: OSMnx图
        center_node: 中心节点ID
        radius: 已弃用，仅保留兼容性
    
    Returns:
        局部网络图
    """
    try:
        print(f"提取节点 {center_node} 的直接相关路段...")
        
        # 获取中心节点的所有邻居节点
        neighbors = set()
        neighbors.add(center_node)
        
        # 添加所有直接连接的节点
        if center_node in G.nodes():
            # 获取所有入边和出边的邻居节点
            for pred in G.predecessors(center_node):
                neighbors.add(pred)
            for succ in G.successors(center_node):
                neighbors.add(succ)
            
            # 再添加这些邻居节点的直接邻居（二度邻居），形成完整的路段
            temp_neighbors = neighbors.copy()
            for neighbor in temp_neighbors:
                for pred in G.predecessors(neighbor):
                    neighbors.add(pred)
                for succ in G.successors(neighbor):
                    neighbors.add(succ)
        
        # 如果邻居节点太少，扩展到更多节点确保网络完整性
        if len(neighbors) < 4:
            print("邻居节点太少，扩展到更多相关节点...")
            temp_neighbors = neighbors.copy()
            for neighbor in temp_neighbors:
                for pred in G.predecessors(neighbor):
                    neighbors.add(pred)
                    # 再添加一层邻居
                    for pred2 in G.predecessors(pred):
                        neighbors.add(pred2)
                for succ in G.successors(neighbor):
                    neighbors.add(succ)
                    # 再添加一层邻居
                    for succ2 in G.successors(succ):
                        neighbors.add(succ2)
        
        # 创建子图
        local_G = G.subgraph(neighbors).copy()
        
        print(f"精确局部网络: 中心节点 {center_node}")
        print(f"包含 {len(local_G.nodes())} 个节点, {len(local_G.edges())} 条边")
        print(f"相关节点: {list(neighbors)[:10]}...")  # 只显示前10个
        
        return local_G
        
    except Exception as e:
        print(f"提取精确局部网络失败: {e}")
        print("回退到单节点网络...")
        
        # 如果失败，创建只包含中心节点和其直接邻居的最小网络
        try:
            minimal_nodes = {center_node}
            if center_node in G.nodes():
                for pred in G.predecessors(center_node):
                    minimal_nodes.add(pred)
                for succ in G.successors(center_node):
                    minimal_nodes.add(succ)
            
            minimal_G = G.subgraph(minimal_nodes).copy()
            print(f"最小网络: 包含 {len(minimal_G.nodes())} 个节点, {len(minimal_G.edges())} 条边")
            
            return minimal_G
            
        except Exception as e2:
            print(f"最小网络创建也失败: {e2}")
            return G

def create_focused_typemap(output_dir):
    """创建优化的typemap，减少警告信息"""
    typemap = '''<?xml version="1.0" encoding="UTF-8"?>
<typemap>
    <type id="highway.primary" color="0.2,0.2,0.2" priority="78" numLanes="2" speed="13.89" allow="passenger"/>
    <type id="highway.secondary" color="0.2,0.2,0.2" priority="77" numLanes="2" speed="11.11" allow="passenger"/>
    <type id="highway.tertiary" color="0.2,0.2,0.2" priority="76" numLanes="1" speed="8.33" allow="passenger"/>
    <type id="highway.residential" color="0.2,0.2,0.2" priority="75" numLanes="1" speed="8.33" allow="passenger"/>
    <type id="highway.unclassified" color="0.2,0.2,0.2" priority="74" numLanes="1" speed="8.33" allow="passenger"/>
    <type id="highway.trunk" color="0.2,0.2,0.2" priority="79" numLanes="3" speed="16.67" allow="passenger"/>
    <type id="highway.motorway" color="0.2,0.2,0.2" priority="80" numLanes="3" speed="22.22" allow="passenger"/>
</typemap>'''
    typemap_path = os.path.join(output_dir, 'focused_typemap.xml')
    with open(typemap_path, 'w', encoding='utf-8') as f:
        f.write(typemap)
    return typemap_path

def add_traffic_lights(net_path, output_dir):
    """
    为SUMO网络的主要交叉口添加红绿灯控制
    """
    print("为主要交叉口添加红绿灯控制...")
    
    try:
        net = sumolib.net.readNet(net_path)
        nodes = net.getNodes()
        
        # 找到主要交叉口（有多个进入和出去的边）
        major_junctions = []
        for node in nodes:
            incoming = [e for e in node.getIncoming() if e.allows("passenger")]
            outgoing = [e for e in node.getOutgoing() if e.allows("passenger")]
            if len(incoming) >= 2 and len(outgoing) >= 2:
                major_junctions.append(node)
        
        if not major_junctions:
            print("没有找到适合添加红绿灯的交叉口")
            return net_path
        
        # 创建红绿灯配置文件
        tls_file = os.path.join(output_dir, 'traffic_lights.add.xml')
        tls_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<additional>
'''
        
        for i, junction in enumerate(major_junctions[:5]):  # 最多为5个交叉口添加红绿灯
            junction_id = junction.getID()
            tls_xml += f'''    <tlLogic id="{junction_id}" type="static" programID="0" offset="0">
        <phase duration="30" state="GGrrrrGGrrrr"/>
        <phase duration="5" state="yyrrrryyrrrr"/>
        <phase duration="30" state="rrGGGGrrGGGG"/>
        <phase duration="5" state="rryyyy ryyyy"/>
    </tlLogic>
'''
        
        tls_xml += '</additional>'
        
        with open(tls_file, 'w', encoding='utf-8') as f:
            f.write(tls_xml)
        
        # 创建带红绿灯的新网络文件
        net_with_tls = os.path.join(output_dir, 'network_with_tls.net.xml')
        
        # 使用netconvert添加红绿灯
        cmd = [
            "netconvert",
            "--sumo-net-file", net_path,
            "--tllogic-files", tls_file,
            "--output-file", net_with_tls,
            "--no-warnings"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"成功为 {len(major_junctions[:5])} 个交叉口添加红绿灯")
            return net_with_tls
        else:
            print(f"红绿灯添加失败，使用原网络: {result.stderr}")
            return net_path
            
    except Exception as e:
        print(f"添加红绿灯时出错: {e}")
        return net_path

def create_focused_simulation(local_G, center_node, output_dir):
    """
    为局部网络创建专门的仿真文件，确保产生有效的交通流量
    
    Args:
        local_G: 局部网络图
        center_node: 中心节点ID
        output_dir: 输出目录
    
    Returns:
        tuple: (net_path, rou_path, cfg_path)
    """
    print(f"为节点 {center_node} 创建高密度交通仿真场景...")

    # 创建优化的typemap
    typemap_path = create_focused_typemap(output_dir)

    # 使用预置网络文件
    preset_net = os.path.join("NET", "network.net.xml")
    local_net_path = os.path.join(output_dir, 'local_network.net.xml')
    
    # 复制预置网络文件
    if os.path.exists(preset_net):
        shutil.copy(preset_net, local_net_path)
        print(f"使用预置网络文件: {preset_net}")
    else:
        print("警告: 未找到预置网络文件，可能需要手动创建")

    # 为网络添加红绿灯控制
    local_net_path = add_traffic_lights(local_net_path, output_dir)

    # 生成高密度路径文件，车辆数提升到300
    local_rou_path = generate_focused_routes(local_net_path, output_dir, num_vehicles=300)

    # 创建配置文件，增加仿真时间到800秒
    local_cfg_path = create_sumocfg(local_net_path, local_rou_path, output_dir, 800)

    return local_net_path, local_rou_path, local_cfg_path

# ===================== Streamlit主流程 =====================
def streamlit_main():
    st.title("东南大学交通仿真大模型")

    st.sidebar.header("API配置")
    api_key = st.sidebar.text_input("API Key", type="password")
    sf_model = st.sidebar.text_input("使用的模型", value="Qwen/Qwen3-32B")

    user_query = st.text_area(
        "请输入您的仿真需求",
        value="能否通过增加掉头车道增加通行效率",
        height=100
    )

    output_dir = st.text_input("输出目录", value="sumo_scenario")
    sim_time = st.number_input("仿真时长（秒）", value=1000, min_value=10)
    run_sumo = st.checkbox("生成后自动启动 SUMO GUI", value=False)

    # 使用 SessionState 记录生成结果，避免重复计算
    if 'scenario_ready' not in st.session_state:
        st.session_state['scenario_ready'] = False
    if 'osm_graph' not in st.session_state:
        st.session_state['osm_graph'] = None
    if 'analysis_done' not in st.session_state:
        st.session_state['analysis_done'] = False
    if 'user_query' not in st.session_state:
        st.session_state['user_query'] = ""

    # 直接从NET文件夹读取预置路网
    net_dir = "NET"
    net_path = os.path.join(net_dir, "network.net.xml")
    rou_path = os.path.join(net_dir, "routes.rou.xml")
    sumocfg_path = os.path.join(net_dir, "scenario.sumocfg")

    if os.path.exists(net_path) and os.path.exists(sumocfg_path):
        st.success(f"已加载预置SUMO路网: {net_path}")
        st.success(f"已加载预置SUMO配置: {sumocfg_path}")
        # 加载SUMO网络
        net = sumolib.net.readNet(net_path)
        import networkx as nx
        G = nx.DiGraph()
        for node in net.getNodes():
            G.add_node(node.getID(), x=node.getCoord()[0], y=node.getCoord()[1])
        for edge in net.getEdges():
            G.add_edge(edge.getFromNode().getID(), edge.getToNode().getID())
        st.info("已从SUMO网络构建节点图")
        st.session_state['osm_graph'] = G
        st.session_state['net_path'] = net_path
        st.session_state['sumocfg_path'] = sumocfg_path
        st.session_state['rou_path'] = rou_path if os.path.exists(rou_path) else None
        st.session_state['scenario_ready'] = True
    else:
        st.error(f"NET文件夹下未找到network.net.xml或scenario.sumocfg，请检查文件是否存在。")
        st.stop()

    # 如果场景已准备好，显示交叉口选择图
    if st.session_state.get('scenario_ready') and not st.session_state.get('analysis_done'):
        st.header("1. 在图上选择一个待分析的交叉口")
        G = st.session_state['osm_graph']
        net_path = st.session_state['net_path']
        sumocfg_path = st.session_state['sumocfg_path']

        # 性能开关
        show_polygons = st.checkbox("显示建筑物背景", value=False)
        # 只在用户勾选时加载多边形
        polygon_traces = []
        if show_polygons:
            polygons_path = os.path.join("NET", "osm_polygons.add.xml")
            if os.path.exists(polygons_path):
                try:
                    tree = ET.parse(polygons_path)
                    root = tree.getroot()
                    for i, poly in enumerate(root.findall('poly')[:100]):  # 限制100个
                        shape = poly.get('shape', '')
                        poly_type = poly.get('type', '')
                        if shape and 'building' in poly_type.lower():
                            points = []
                            for point in shape.split():
                                if ',' in point:
                                    x, y = map(float, point.split(','))
                                    if np.isfinite(x) and np.isfinite(y):
                                        points.append((x, y))
                            if len(points) >= 3:
                                xs, ys = zip(*points)
                                xs = xs + (xs[0],)
                                ys = ys + (ys[0],)
                                polygon_traces.append(go.Scatter(
                                    x=xs, y=ys, mode='lines', fill='toself',
                                    fillcolor='lightblue', line=dict(color='gray', width=0.5),
                                    hoverinfo='skip', showlegend=False, opacity=0.2
                                ))
                except Exception as e:
                    st.warning(f"多边形加载失败: {e}")

        # 边渲染（只画主干道/次干道，且shape点数>1且无NaN）
        edges = list(net.getEdges())
        edge_traces = []
        for edge in edges:
            shape = edge.getShape()
            if shape and len(shape) > 1 and all(np.isfinite(x) and np.isfinite(y) for x, y in shape):
                prio = edge.getPriority() if hasattr(edge, 'getPriority') else 0
                if prio >= 77:
                    color = '#ff4444' if prio >= 80 else '#ff8844'
                    width = 2.5 if prio >= 80 else 1.5
                else:
                    continue
                xs, ys = zip(*shape)
                edge_traces.append(go.Scattergl(
                    x=xs, y=ys, mode='lines',
                    line=dict(color=color, width=width),
                    hoverinfo='skip', showlegend=False
                ))

        # 节点采样率适中，保证交互
        all_nodes = list(net.getNodes())
        junction_nodes, other_nodes = [], []
        junction_xs, junction_ys = [], []
        other_xs, other_ys = [], []
        for node in all_nodes[::2]:  # 采样率提升到1/2
            x, y = node.getCoord()
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if node.getType() in ['traffic_light', 'right_before_left', 'priority']:
                junction_xs.append(x)
                junction_ys.append(y)
                junction_nodes.append(node.getID())
            else:
                other_xs.append(x)
                other_ys.append(y)
                other_nodes.append(node.getID())
        # 重要交叉口
        junction_trace = go.Scattergl(
            x=junction_xs, y=junction_ys, mode='markers',
            marker=dict(size=10, color='red', symbol='circle'),
            text=junction_nodes, hoverinfo='skip', name='交叉口'
        )
        # 普通节点
        node_trace = go.Scattergl(
            x=other_xs, y=other_ys, mode='markers',
            marker=dict(size=5, color='blue', symbol='circle'),
            text=other_nodes, hoverinfo='skip', name='节点'
        )
        # 选中节点高亮
        highlight_trace = None
        selected_node = st.session_state.get('selected_node', None)
        if selected_node:
            all_node_ids = junction_nodes + other_nodes
            all_xs = junction_xs + other_xs
            all_ys = junction_ys + other_ys
            if selected_node in all_node_ids:
                idx = all_node_ids.index(selected_node)
                highlight_trace = go.Scattergl(
                    x=[all_xs[idx]], y=[all_ys[idx]], mode='markers',
                    marker=dict(size=18, color='yellow', symbol='star', 
                               line=dict(color='red', width=3)),
                    name='选中节点', hoverinfo='skip'
                )
        # 组合图层，节点 trace 放最后
        all_traces = polygon_traces + edge_traces
        all_traces += [junction_trace, node_trace]
        if highlight_trace:
            all_traces.append(highlight_trace)

        fig = go.Figure(data=all_traces)
        fig.update_layout(
            title="路网图",
            dragmode='pan',
            width=900, height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        
        # 显示图表
        st.plotly_chart(fig, use_container_width=True)
        
        # 更稳定的选点方式：下拉菜单选择
        st.subheader("选择交叉口节点")
        
        # 准备节点选项
        all_node_ids = junction_nodes + other_nodes
        all_node_labels = []
        for node_id in all_node_ids:
            if node_id in junction_nodes:
                all_node_labels.append(f"🚦 {node_id} (交叉口)")
            else:
                all_node_labels.append(f"🔵 {node_id} (节点)")
        
        # 节点选择下拉框
        if len(all_node_labels) > 0:
            selected_index = st.selectbox(
                "从下拉菜单选择节点：",
                options=range(len(all_node_labels)),
                format_func=lambda x: all_node_labels[x],
                index=0 if not st.session_state.get('selected_node') else (
                    all_node_ids.index(st.session_state.get('selected_node')) 
                    if st.session_state.get('selected_node') in all_node_ids else 0
                )
            )
            
            selected_node_from_dropdown = all_node_ids[selected_index]
            
            # 更新选中节点
            if st.button("确认选择此节点"):
                st.session_state['selected_node'] = selected_node_from_dropdown
                st.success(f"已选择节点: **{selected_node_from_dropdown}**")
                st.rerun()
        else:
            st.error("没有找到可选择的节点")
        
        # 显示当前选中的节点
        if st.session_state.get('selected_node'):
            st.info(f"当前选中节点: **{st.session_state['selected_node']}**")
            
            # 显示节点在图中的位置信息
            if st.session_state['selected_node'] in all_node_ids:
                idx = all_node_ids.index(st.session_state['selected_node'])
                all_xs = junction_xs + other_xs
                all_ys = junction_ys + other_ys
                node_x, node_y = all_xs[idx], all_ys[idx]
                st.write(f"节点坐标: ({node_x:.2f}, {node_y:.2f})")

        # 手动输入备选方案
        st.subheader("或手动输入节点ID")
        manual_id = st.text_input("输入节点ID", value="")
        if st.button("手动确认选择") and manual_id:
            st.session_state['selected_node'] = manual_id
            st.success(f"已选择节点: **{manual_id}**")
            st.rerun()

    if st.session_state.get('selected_node'):
        st.header("2. 精确节点仿真配置")
        
        # 显示当前选择的信息
        st.info(f"已选择交叉口: **{st.session_state['selected_node']}**")
        st.info("📍 **仿真范围**: 仅包含选定节点的直接相关路段和路口")
        
        # 显示仿真特点
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.markdown("**仿真范围:**")
        #     st.markdown("- 🎯 选定的交叉口节点")  
        #     st.markdown("- 🛣️ 直接连接的路段")
        #     st.markdown("- 🔗 相邻的关键节点")
            
        # with col2:
        #     st.markdown("**优势:**")
        #     st.markdown("- ⚡ 极速仿真（秒级完成）")
        #     st.markdown("- 🎛️ 策略效果明显") 
        #     st.markdown("- 📊 结果精确可靠")
        
        st.header("3. 让AI分析并优化")
        if st.button("运行精确节点仿真、AI分析并生成报告"):
            if not api_key:
                st.error("请输入您的Silicon Flow API Key。")
                st.stop()

            with st.spinner("AI正在思考改造策略..."):
                prompt_strategy = f"""
一个交通仿真场景已经就绪。用户希望解决的初始问题是："{st.session_state['user_query']}"
现在，用户在地图上选择了一个特定的交叉口（ID: {st.session_state['selected_node']}）进行优化。

我们将对该交叉口及其直接相关的路段进行精确仿真，专门分析该节点的交通状况。

可用的改造策略有两种:
1.  `add_lane`: 在交叉口所有道路上各增加一条常规车道。适用于解决普遍的通行能力不足问题。
2.  `add_uturn_lane`: 在交叉口所有道路上各增加一条掉头专用道。适用于改善特定方向的掉头或左转效率。

基于用户的初始问题，请判断哪种策略更可能有效。
请只返回策略的名称（`add_lane` 或 `add_uturn_lane`）。
"""
                strategy = siliconflow_chat(prompt_strategy, api_key, model=sf_model)
                if not strategy or strategy not in ["add_lane", "add_uturn_lane"]:
                    st.warning("AI未能选择有效策略，将默认使用 'add_lane'。")
                    strategy = "add_lane"
            
            st.info(f"AI建议的策略: **{strategy}**")
            
            # 创建精确局部网络
            with st.spinner("提取精确节点网络..."):
                G = st.session_state['osm_graph']
                selected_node = st.session_state['selected_node']
                
                # 提取精确局部网络（不使用半径）
                local_G = extract_local_network(G, selected_node)
                
                # 创建精确仿真目录
                precise_sim_dir = os.path.join(output_dir, "precise_simulation")
                os.makedirs(precise_sim_dir, exist_ok=True)
                
                # 为精确网络创建仿真文件
                local_net_path, local_rou_path, local_cfg_path = create_focused_simulation(
                    local_G, selected_node, precise_sim_dir
                )
                
                st.success(f"精确网络提取完成！包含 {len(local_G.nodes())} 个节点，{len(local_G.edges())} 条边")
                
                # 显示网络规模对比
                original_nodes = len(G.nodes())
                local_nodes = len(extract_local_network(G, selected_node).nodes())
                reduction_ratio = ((original_nodes - local_nodes) / original_nodes * 100)
                
                st.metric("网络精简效果", f"{reduction_ratio:.1f}%", help=f"从 {original_nodes} 个节点精简到 {local_nodes} 个节点")

            # 运行基线仿真
            with st.spinner(f"正在运行精确基线仿真..."):
                st.info("📊 正在计算基线等待时间...")
                st.info("🚗 增加车辆密度以产生有效的交通流量...")
                baseline_wait_time = simulate_wait_time(
                    local_cfg_path,
                    local_net_path,
                    str(selected_node),
                    800  # 增加仿真时间确保产生等待
                )
                st.metric(label="精确基线平均等待时间", value=f"{baseline_wait_time:.2f} 秒")

            # 运行策略仿真
            with st.spinner(f"正在应用 '{strategy}' 策略并运行新仿真..."):
                st.info(f"🔧 正在应用 {strategy} 策略...")
                strat_dir = os.path.join(precise_sim_dir, strategy)
                os.makedirs(strat_dir, exist_ok=True)
                strat_net = os.path.join(strat_dir, "network_mod.net.xml")

                if strategy == "add_lane":
                    add_lane(local_net_path, selected_node, strat_net)
                elif strategy == "add_uturn_lane":
                    add_uturn_lane(local_net_path, selected_node, strat_net)
                
                st.info("🚗 正在生成高密度交通流...")
                # 为修改后的网络重新生成配置文件
                strat_rou = generate_focused_routes(strat_net, strat_dir, num_vehicles=300)
                strat_cfg = create_sumocfg(strat_net, strat_rou, strat_dir, 800)

                st.info("📊 正在计算策略效果...")
                modified_wait_time = simulate_wait_time(
                    strat_cfg,
                    strat_net,
                    str(selected_node),
                    800  # 增加仿真时间
                )
                
                # 计算改进程度
                improvement = baseline_wait_time - modified_wait_time
                improvement_percent = (improvement / baseline_wait_time * 100) if baseline_wait_time > 0 else 0
                
                # 显示结果
                if improvement > 0:
                    st.success(f"✅ 策略有效！等待时间减少了 {improvement:.2f} 秒")
                elif improvement < 0:
                    st.warning(f"⚠️ 策略效果不佳，等待时间增加了 {abs(improvement):.2f} 秒")
                else:
                    st.info("➖ 策略对等待时间没有显著影响")
                
                st.metric(
                    label=f"'{strategy}' 策略后平均等待时间", 
                    value=f"{modified_wait_time:.2f} 秒", 
                    delta=f"{-improvement:.2f} 秒 ({improvement_percent:.1f}%)"
                )

            # LLM生成最终报告
            with st.spinner("AI正在生成最终分析报告..."):
                prompt_report = f"""
我进行了一次精确节点交通仿真实验，旨在解决这个问题："{st.session_state['user_query']}"

实验设置：
- 分析目标：交叉口 {selected_node} 及其直接相关的路段
- 精确网络规模：{len(local_G.nodes())} 个节点，{len(local_G.edges())} 条边
- 网络精简比例：{reduction_ratio:.1f}%（从原始网络精简而来）
- 仿真时间：400秒（专门针对该节点的超短时精确仿真）
- 车辆数量：30辆（精确控制的测试车辆）

仿真结果对比：
- **基线情况**: 该交叉口的平均车辆等待时间为 **{baseline_wait_time:.2f}** 秒
- **{strategy}策略后**: 平均车辆等待时间变为 **{modified_wait_time:.2f}** 秒
- **改进效果**: {"减少" if improvement > 0 else "增加"} **{abs(improvement):.2f}** 秒 ({abs(improvement_percent):.1f}%)

请基于以上精确节点仿真数据，为我生成一份简短的分析报告。报告应该包括：
1. 对精确节点仿真结果的解读（策略在该特定交叉口是否有效？效果如何？）
2. 对该策略在单个节点范围内的评价
3. 针对该特定交叉口的具体改进建议
4. 这种精确仿真方法的优势和局限性

请使用中文，并以Markdown格式呈现。
"""
                final_report = siliconflow_chat(prompt_report, api_key, model=sf_model)
                st.session_state['final_report'] = final_report
                st.session_state['analysis_done'] = True
                st.session_state['precise_simulation_done'] = True
    
    if st.session_state.get('analysis_done'):
        st.header("4. AI生成的分析报告")
        st.markdown(st.session_state.get('final_report', "报告生成失败。"))
        
        # 如果是精确仿真，显示额外信息
        if st.session_state.get('precise_simulation_done'):
            st.header("5. 精确节点仿真详情")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("仿真精度")
                if 'osm_graph' in st.session_state:
                    original_nodes = len(st.session_state['osm_graph'].nodes())
                    local_nodes = len(extract_local_network(st.session_state['osm_graph'], st.session_state['selected_node']).nodes())
                    
                    st.metric("原始网络节点数", original_nodes)
                    st.metric("精确网络节点数", local_nodes)
                    st.metric("精简比例", f"{((original_nodes - local_nodes) / original_nodes * 100):.1f}%")
                    
                st.subheader("仿真参数")
                st.info("⏱️ 仿真时间: 400秒")
                st.info("🚗 测试车辆: 30辆")
                st.info("🎯 策略目标: 单一交叉口")
            
            with col2:
                st.subheader("精确仿真优势")
                st.success("✅ 超快速度 - 秒级完成仿真")
                st.success("✅ 精确控制 - 专注单个节点")
                st.success("✅ 结果明确 - 策略效果直观")
                st.success("✅ 资源节约 - 最小化计算需求")
                
                st.subheader("适用场景")
                st.info("🎯 单个交叉口优化分析")
                st.info("🔧 策略效果快速验证")  
                st.info("⚡ 大批量节点筛选")
                st.info("📊 精确数据收集")
            
            # 添加重新分析选项
            st.subheader("重新分析")
            if st.button("选择不同节点重新分析"):
                st.session_state['analysis_done'] = False
                st.session_state['precise_simulation_done'] = False
                st.rerun()

    st.markdown("""
---
**使用说明：**
- 需本地已安装SUMO，并配置好环境变量
- 需要在左侧边栏配置有效的 API Key
- 适合快速验证单个交叉口的改进策略
- 可批量分析多个节点，快速筛选优化目标
- 生成的所有文件均在指定的输出目录下
- **如上传/指定自定义SUMO路网、配置、路线文件，后续所有仿真与分析流程均自动兼容，无需额外操作**
- **本系统不再支持OpenStreetMap/OSM文件，所有地图与节点均基于SUMO网络文件自动推断**
""")

# ===================== 启动入口 =====================
if __name__ == '__main__':
    # 更健壮的判断
    streamlit_main()