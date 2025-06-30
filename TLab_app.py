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
        st.error(f"调用Silicon Flow API失败: {e}")
        return None
    except (KeyError, IndexError) as e:
        st.error(f"解析API响应失败: {e}")
        st.error(f"收到的响应: {response.text}")
        return None

# ===================== 核心功能函数 =====================
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
        node_obj = net.getNode(str(junction_id))
        if node_obj is None:
            raise RuntimeError(f"在 SUMO 网络中未找到节点 {junction_id}")
        edges_of_junction = set([e.getID() for e in node_obj.getIncoming()])
        edges_of_junction.update([e.getID() for e in node_obj.getOutgoing()])

    # 启动 TraCI 仿真
    sumo_binary = 'sumo'
    if os.environ.get('SUMO_GUI') == '1':
        sumo_binary = 'sumo-gui'
    traci.start([sumo_binary, '-c', sumocfg_path, '--step-length', '1'])

    total_wait = 0
    count = 0
    for _ in range(sim_time):
        traci.simulationStep()
        for vid in traci.vehicle.getIDList():
            edge_id = traci.vehicle.getRoadID(vid)
            if edges_of_junction is None or edge_id in edges_of_junction:
                total_wait += traci.vehicle.getWaitingTime(vid)
                count += 1
    traci.close()
    return total_wait / count if count else 0

# ==== 网络改造策略：在指定交叉口增加掉头车道 ====
def add_uturn_lane(net_path, junction_id, output_net_path):
    tree = ET.parse(net_path)
    root = tree.getroot()
    for edge in root.findall('edge'):
        if edge.get('function') == 'internal':
            continue
        if edge.get('from') == str(junction_id) or edge.get('to') == str(junction_id):
            lanes = edge.findall('lane')
            if not lanes:
                continue
            last_lane = lanes[-1]
            new_index = len(lanes)

            # 复制属性字典
            new_attrs = last_lane.attrib.copy()
            new_attrs.update({
                'id': f"{edge.get('id')}_{new_index}",
                'index': str(new_index)
            })
            new_lane = ET.Element('lane', new_attrs)
            edge.append(new_lane)

            # 更新 edge 的 lane 数（如果存在）
            if 'numLanes' in edge.attrib:
                edge.set('numLanes', str(int(edge.get('numLanes')) + 1))
    tree.write(output_net_path)
    return output_net_path

# ==== 网络改造策略：在指定交叉口关联道路增加 1 条普通车道 ====
def add_lane(net_path, junction_id, output_net_path):
    """
    在与指定节点直接相连的所有 edge 上增加 1 条同属性车道，
    并用 netconvert 重新生成 connection，最终保存到 output_net_path
    """
    # 1. 复制原 net.xml 后直接改写 lane
    temp_net = tempfile.mktemp(suffix=".net.xml")
    shutil.copy(net_path, temp_net)

    tree = ET.parse(temp_net)
    root = tree.getroot()
    modified_edges = {}
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        if edge.get("from") == str(junction_id) or edge.get("to") == str(junction_id):
            lanes = edge.findall("lane")
            if not lanes:
                continue
            last_lane = lanes[-1]
            new_index = len(lanes)
            new_lane = ET.Element("lane", last_lane.attrib.copy())
            # 更新 id / index
            new_lane.set("id", f"{edge.get('id')}_{new_index}")
            new_lane.set("index", str(new_index))
            edge.append(new_lane)
            # 同步 numLanes（如果有）
            if "numLanes" in edge.attrib:
                edge.set("numLanes", str(int(edge.get("numLanes")) + 1))
            modified_edges[edge.get('id')] = new_index

    # 复制 connection 元素以连接新增车道（必须在写文件之前）
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

    # 2. 让 netconvert 重建 connection (捕获输出便于调试)
    cmd = [
        "netconvert",
        "--sumo-net-file", temp_net,
        "--output-file", output_net_path
    ]
    st.write("Rebuilding network with: ", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        st.error("netconvert 失败，输出如下：")
        st.code(result.stderr)
        # 回退：直接使用修改后的 temp_net（未重建 connection），以便继续调试
        shutil.copy(temp_net, output_net_path)
        st.warning("已回退到未重建 connection 的网络文件，可能缺少连接！")

    # 3. 清理
    os.remove(temp_net)

    return output_net_path

# ===================== Streamlit主流程 =====================
def streamlit_main():
    st.title("东南大学交通仿真大模型")

    # st.info("现在我想要加入ollama大模型的帮助，帮助我调用仿真，包括语义理解在哪个城市的哪个区域，与openstreetmap交互，然后运行仿真添加策略也是大模型思考得到的，最后由大模型思考仿真结果生成最终方案")

    st.sidebar.header("API配置")
    api_key = st.sidebar.text_input("API Key", type="password")
    sf_model = st.sidebar.text_input("使用的Silicon Flow模型", value="Qwen/Qwen3-32B")

    user_query = st.text_area(
        "请输入您的仿真需求",
        "我想分析一下东南大学的交通状况，看看能不能通过增加车道来改善拥堵。",
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

    if st.button("让AI分析需求并生成场景"):
        st.session_state['scenario_ready'] = False
        st.session_state['analysis_done'] = False
        st.session_state['user_query'] = user_query

        if not api_key:
            st.error("请输入您的Silicon Flow API Key。")
            st.stop()

        with st.spinner("请求AI分析地理位置..."):
            # 1. 让 LLM 提取地名
            prompt_place = f"""
从以下用户查询中，提取用于OpenStreetMap的地理位置名称。
请只返回最具体的地点、城市和国家，格式为 "地点, 城市, 国家"。
例如，对于 "我想看看南京市中心新街口地区的交通"，你应该返回 "Xinjiekou, Nanjing, China"。
对于 "分析一下美国旧金山的交通状况"，你应该返回 "San Francisco, CA, USA"。
请务必包含国家，以提高准确性。
用户查询: "{user_query}"
"""
            place = siliconflow_chat(prompt_place, api_key, model=sf_model)

            if not place:
                st.error("未能从AI获取有效的地理位置。")
                st.stop()
        
        st.info(f"AI识别出的地点: **{place}**")

        os.makedirs(output_dir, exist_ok=True)
        with st.spinner("下载OSM数据..."):
            try:
                G = ox.graph_from_place(place, network_type='drive', simplify=False)
            except Exception as e:
                st.error(f"从OSM下载数据失败: {e}")
                st.error("请检查AI识别的地点是否准确，或尝试更具体的描述。")
                st.stop()
            osm_path = os.path.join(output_dir, 'network.osm')
            ox.save_graph_xml(G, osm_path)
            st.session_state['osm_graph'] = G
            st.success("OSM数据下载完成！")
        with st.spinner("生成typemap.xml..."):
            typemap_path = create_typemap(output_dir)
            st.success("typemap.xml生成完成！")
        with st.spinner("OSM转SUMO网络..."):
            net_path = osm_to_sumo(osm_path, typemap_path, output_dir)
            st.success("SUMO网络文件生成完成！")
        with st.spinner("生成随机路线..."):
            rou_path = generate_routes(net_path, output_dir, sim_time)
            st.success("路线文件生成完成！")
        with st.spinner("生成SUMO配置文件..."):
            sumocfg_path = create_sumocfg(net_path, rou_path, output_dir, sim_time)
            st.success("SUMO配置文件生成完成！")
        st.success(f"SUMO场景已生成于 {output_dir}")
        st.session_state['scenario_ready'] = True
        st.session_state['net_path'] = net_path
        st.session_state['sumocfg_path'] = sumocfg_path
        st.session_state['rou_path'] = rou_path
        if run_sumo:
            st.info("正在启动SUMO GUI...")
            subprocess.Popen(['sumo-gui', '-c', sumocfg_path])

    # 如果场景已准备好，显示交叉口选择地图
    if st.session_state.get('scenario_ready') and not st.session_state.get('analysis_done'):
        st.header("1. 在地图上选择一个待分析的交叉口")
        G = st.session_state['osm_graph']
        net_path = st.session_state['net_path']
        sumocfg_path = st.session_state['sumocfg_path']

        # 构建 Folium 地图
        center_y, center_x = list(G.nodes(data='y'))[0][1], list(G.nodes(data='x'))[0][1]
        m = folium.Map(location=[center_y, center_x], zoom_start=16)

        for nid, data in G.nodes(data=True):
            folium.CircleMarker(location=(data['y'], data['x']), radius=3, tooltip=str(nid)).add_to(m)
        map_data = st_folium(m, height=500, width=700)

        selected_node = None
        if map_data and map_data.get('last_clicked'):
            lat = map_data['last_clicked']['lat']
            lon = map_data['last_clicked']['lng']
            selected_node = ox.distance.nearest_nodes(G, lon, lat)
            st.session_state['selected_node'] = selected_node
            st.info(f"已选择交叉口节点 ID: **{selected_node}**")

    if st.session_state.get('selected_node'):
        st.header("2. 让AI分析并优化")
        if st.button("运行仿真、AI分析并生成报告"):
            if not api_key:
                st.error("请输入您的Silicon Flow API Key。")
                st.stop()

            with st.spinner("AI正在思考改造策略..."):
                prompt_strategy = f"""
一个交通仿真场景已经就绪。用户希望解决的初始问题是："{st.session_state['user_query']}"
现在，用户在地图上选择了一个特定的交叉口（ID: {st.session_state['selected_node']}）进行优化。

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

            # 运行基线仿真
            with st.spinner(f"正在运行 Baseline 仿真..."):
                baseline_wait_time = simulate_wait_time(
                    st.session_state['sumocfg_path'],
                    st.session_state['net_path'],
                    str(st.session_state['selected_node']),
                    int(sim_time)
                )
                st.metric(label="Baseline 平均等待时间", value=f"{baseline_wait_time:.2f} 秒")

            # 运行策略仿真
            with st.spinner(f"正在应用 '{strategy}' 策略并运行新仿真..."):
                strat_dir = os.path.join(output_dir, strategy)
                os.makedirs(strat_dir, exist_ok=True)
                strat_net = os.path.join(strat_dir, "network_mod.net.xml")

                if strategy == "add_lane":
                    add_lane(st.session_state['net_path'], st.session_state['selected_node'], strat_net)
                elif strategy == "add_uturn_lane":
                    add_uturn_lane(st.session_state['net_path'], st.session_state['selected_node'], strat_net)
                
                # 为修改后的网络重新生成配置文件
                strat_rou = generate_routes(strat_net, strat_dir, sim_time)
                strat_cfg = create_sumocfg(strat_net, strat_rou, strat_dir, sim_time)

                modified_wait_time = simulate_wait_time(
                    strat_cfg,
                    strat_net,
                    str(st.session_state['selected_node']),
                    int(sim_time)
                )
                st.metric(label=f"'{strategy}' 策略后平均等待时间", value=f"{modified_wait_time:.2f} 秒", delta=f"{modified_wait_time - baseline_wait_time:.2f} 秒")

            # LLM生成最终报告
            with st.spinner("AI正在生成最终分析报告..."):
                prompt_report = f"""
我进行了一次交通仿真实验，旨在解决这个问题："{st.session_state['user_query']}"

在一个特定的交叉口（ID: {st.session_state['selected_node']}），我进行了如下对比：
- **基线情况 (Baseline)**: 交叉口的平均车辆等待时间为 **{baseline_wait_time:.2f}** 秒。
- **改造后**: 我采纳了你的建议，应用了 `{strategy}` 策略。改造后，该交叉口的平均车辆等待时间变为 **{modified_wait_time:.2f}** 秒。

请基于以上数据，为我生成一份简短的分析报告。报告应该包括：
1.  对仿真结果的解读（策略是否有效？效果如何？）。
2.  对该策略的评价。
3.  未来的建议。

请使用中文，并以Markdown格式呈现。
"""
                final_report = siliconflow_chat(prompt_report, api_key, model=sf_model)
                st.session_state['final_report'] = final_report
                st.session_state['analysis_done'] = True
    
    if st.session_state.get('analysis_done'):
        st.header("3. AI生成的分析报告")
        st.markdown(st.session_state.get('final_report', "报告生成失败。"))

    st.markdown("""
---
**提示：**
- 需本地已安装SUMO，并配置好环境变量（netconvert、randomTrips.py、sumo-gui等命令可直接调用）。
- 需要在左侧边栏配置有效的 API Key。
- 生成的所有文件均在你指定的输出目录下。
""")

# ===================== 启动入口 =====================
if __name__ == '__main__':
    # 更健壮的判断
    streamlit_main()