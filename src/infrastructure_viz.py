import os
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import pandas as pd
import json

# Configuration
INFRASTRUCTURE_FILE = "generated_infrastructure.pl"
REPORT_DIR = "reports"
OUTPUT_DIR = "visualization_results"

def parse_infrastructure_file(file_path):
    """Parse the generated_infrastructure.pl file to extract nodes and links"""
    if not os.path.exists(file_path):
        print(f"Infrastructure file not found at {file_path}")
        return None, None
    
    print(f"Parsing infrastructure file: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract nodes
    nodes = {}
    node_pattern = r'node\(([\w\d]+),\s*([\w\d]+),\s*([\w\d]+),\s*\[(.*?)\],\s*\[(.*?)\],\s*\((.*?)\)\)'
    node_matches = re.finditer(node_pattern, content)
    
    for match in node_matches:
        node_id = match.group(1)
        node_type = match.group(2)
        provider = match.group(3)
        security_caps = [cap.strip() for cap in match.group(4).split(',')] if match.group(4) else []
        software_caps = [cap.strip() for cap in match.group(5).split(',')] if match.group(5) else []
        
        hw_caps = match.group(6).split(',')
        memory = hw_caps[0].strip()
        vcpu = hw_caps[1].strip() if len(hw_caps) > 1 else "N/A"
        mhz = hw_caps[2].strip() if len(hw_caps) > 2 else "N/A"
        
        nodes[node_id] = {
            'type': node_type,
            'provider': provider,
            'security_caps': security_caps,
            'software_caps': software_caps,
            'memory': memory,
            'vcpu': vcpu,
            'mhz': mhz
        }
    
    print(f"Found {len(nodes)} nodes")
    
    # Extract links
    links = []
    latency_pattern = r'latency\(([\w\d]+),\s*([\w\d]+),\s*(\d+)\)'
    latency_matches = re.finditer(latency_pattern, content)
    
    for match in latency_matches:
        node1 = match.group(1)
        node2 = match.group(2)
        latency = int(match.group(3))
        
        # Skip self-links
        if node1 != node2:
            links.append((node1, node2, latency))
    
    print(f"Found {len(links)} links")
    
    # Extract services
    services = {}
    service_pattern = r'service\(([\w\d]+),\s*([\w\d]+),\s*([\w\d]+),\s*([\w\d]+)\)'
    service_matches = re.finditer(service_pattern, content)
    
    for match in service_matches:
        service_id = match.group(1)
        provider = match.group(2)
        service_type = match.group(3)
        deployed_node = match.group(4)
        
        services[service_id] = {
            'provider': provider,
            'type': service_type,
            'deployed_node': deployed_node
        }
    
    print(f"Found {len(services)} services")
    
    return nodes, links, services

def create_infrastructure_graph(nodes, links):
    """Create a networkx graph from the parsed infrastructure"""
    G = nx.Graph()
    
    # Add nodes with attributes
    for node_id, attrs in nodes.items():
        G.add_node(node_id, **attrs)
    
    # Add edges with latency as weight
    for node1, node2, latency in links:
        G.add_edge(node1, node2, weight=latency, latency=latency)
    
    return G

def find_latest_report():
    """Find the most recent report file in the reports directory"""
    reports_dir = os.path.join(os.getcwd(), REPORT_DIR)
    if not os.path.exists(reports_dir):
        print(f"Reports directory not found at {reports_dir}")
        return None
    
    report_files = [f for f in os.listdir(reports_dir) if f.startswith("report ") and f.endswith(".json")]
    if not report_files:
        print("No report files found")
        return None
    
    # Sort by creation time
    report_files.sort(key=lambda x: os.path.getctime(os.path.join(reports_dir, x)), reverse=True)
    return os.path.join(reports_dir, report_files[0])

def load_report(report_path=None):
    """Load the simulation report from a JSON file"""
    if report_path is None:
        report_path = find_latest_report()
        if report_path is None:
            return None
    
    try:
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        print(f"Loaded report from {report_path}")
        return report_data
    except Exception as e:
        print(f"Error loading report: {e}")
        return None

def plot_infrastructure(G, output_file="infrastructure_network.png"):
    """Create a visualization of the infrastructure network"""
    plt.figure(figsize=(18, 12))
    
    # Set node positions using spring layout
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    
    # Prepare node colors based on type
    node_colors = []
    for node in G.nodes():
        if G.nodes[node]['type'] == 'cloud':
            node_colors.append('#36AE7C')  # Green for cloud
        elif G.nodes[node]['type'] == 'fog':
            node_colors.append('#F9D923')  # Yellow for fog
        else:  # edge
            node_colors.append('#EB5353')  # Red for edge
    
    # Prepare node sizes based on resources
    node_sizes = []
    for node in G.nodes():
        memory = G.nodes[node]['memory']
        if memory == 'inf':
            node_sizes.append(800)  # Large for infinite memory
        else:
            try:
                mem_val = int(memory)
                node_sizes.append(300 + mem_val/10)  # Scale based on memory
            except:
                node_sizes.append(300)  # Default size
    
    # Prepare edge widths based on latency (inverse - thicker for lower latency)
    edge_widths = []
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        latency = data.get('latency', 0)
        edge_widths.append(3 - (latency / 40))  # Thicker lines for lower latency
        edge_labels[(u, v)] = f"{latency}ms"
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
    
    # Draw edge labels (latencies)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Add a legend
    cloud_patch = mpatches.Patch(color='#36AE7C', label='Cloud Node')
    fog_patch = mpatches.Patch(color='#F9D923', label='Fog Node')
    edge_patch = mpatches.Patch(color='#EB5353', label='Edge Node')
    plt.legend(handles=[cloud_patch, fog_patch, edge_patch], loc='upper right')
    
    plt.title('LambdaFogSim Infrastructure Network', fontsize=16)
    plt.tight_layout()
    plt.axis('off')
    
    # Save the figure
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Infrastructure network visualization saved to {output_file}")

def plot_security_levels(G, output_file="security_levels.png"):
    """Create a visualization of security levels in the infrastructure"""
    plt.figure(figsize=(18, 12))
    
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    
    # Categorize nodes by security level
    top_security = []  # Both antiTamp and pubKeyE
    medium_security = []  # pubKeyE only
    low_security = []  # None
    
    for node in G.nodes():
        security_caps = G.nodes[node]['security_caps']
        if 'antiTamp' in security_caps and 'pubKeyE' in security_caps:
            top_security.append(node)
        elif 'pubKeyE' in security_caps:
            medium_security.append(node)
        else:
            low_security.append(node)
    
    # Draw nodes with different colors based on security level
    nx.draw_networkx_nodes(G, pos, nodelist=top_security, node_color='#98FB98', node_size=500, label='Top Security')
    nx.draw_networkx_nodes(G, pos, nodelist=medium_security, node_color='#FFCC99', node_size=400, label='Medium Security')
    nx.draw_networkx_nodes(G, pos, nodelist=low_security, node_color='#FFC0CB', node_size=300, label='Low Security')
    
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    
    plt.title('Infrastructure Security Levels', fontsize=16)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Security levels visualization saved to {output_file}")

def plot_software_capabilities(G, output_file="software_capabilities.png"):
    """Create a visualization showing the distribution of software capabilities"""
    # Count the frequency of each software capability
    software_counts = defaultdict(int)
    node_software = {}
    
    for node in G.nodes():
        software_list = G.nodes[node]['software_caps']
        node_software[node] = software_list
        for software in software_list:
            software_counts[software] += 1
    
    # Create a bar chart of software frequencies
    plt.figure(figsize=(12, 6))
    software_names = list(software_counts.keys())
    frequencies = [software_counts[sw] for sw in software_names]
    
    plt.bar(software_names, frequencies, color=['#3498db', '#2ecc71', '#f1c40f', '#e74c3c'])
    plt.title('Software Capability Distribution in the Infrastructure', fontsize=16)
    plt.xlabel('Software Type')
    plt.ylabel('Number of Nodes')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create another visualization showing software combinations
    plt.figure(figsize=(15, 10))
    
    # Generate unique colors for different node types
    node_type_colors = {'cloud': '#36AE7C', 'fog': '#F9D923', 'edge': '#EB5353'}
    
    # Create a scatter plot with node_type on x-axis and different software combinations on y-axis
    combinations = defaultdict(list)
    
    for node in G.nodes():
        node_type = G.nodes[node]['type']
        sw_combo = ", ".join(sorted(G.nodes[node]['software_caps']))
        combinations[sw_combo].append(node_type)
    
    # Count combinations by node type
    combo_counts = {}
    for combo in combinations:
        combo_counts[combo] = {
            'cloud': combinations[combo].count('cloud'),
            'fog': combinations[combo].count('fog'),
            'edge': combinations[combo].count('edge')
        }
    
    # Convert to DataFrame for plotting
    df_data = []
    for combo, counts in combo_counts.items():
        for node_type, count in counts.items():
            if count > 0:
                df_data.append({
                    'Software Combination': combo,
                    'Node Type': node_type,
                    'Count': count
                })
    
    df = pd.DataFrame(df_data)
    
    # Plot as grouped bar chart
    sw_combinations = sorted(list(combinations.keys()))
    fig, ax = plt.subplots(figsize=(15, 8))
    
    bar_width = 0.25
    index = range(len(sw_combinations))
    
    cloud_counts = [combo_counts[combo].get('cloud', 0) for combo in sw_combinations]
    fog_counts = [combo_counts[combo].get('fog', 0) for combo in sw_combinations]
    edge_counts = [combo_counts[combo].get('edge', 0) for combo in sw_combinations]
    
    ax.bar([i - bar_width for i in index], cloud_counts, bar_width, label='Cloud', color='#36AE7C')
    ax.bar(index, fog_counts, bar_width, label='Fog', color='#F9D923')
    ax.bar([i + bar_width for i in index], edge_counts, bar_width, label='Edge', color='#EB5353')
    
    ax.set_xlabel('Software Combination')
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Software Combinations by Node Type')
    ax.set_xticks(index)
    ax.set_xticklabels(sw_combinations, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file.replace('.png', '_by_node_type.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Software capabilities visualizations saved to {output_file} and {output_file.replace('.png', '_by_node_type.png')}")

def plot_services_distribution(G, services, output_file="services_distribution.png"):
    """Create a visualization of services deployment across the infrastructure"""
    if not services:
        print("No services data available")
        return
    
    # Count services by node type
    services_by_node_type = defaultdict(lambda: defaultdict(int))
    services_by_type = defaultdict(int)
    
    for service_id, service_data in services.items():
        service_type = service_data['type']
        deployed_node = service_data['deployed_node']
        
        # Skip if the deployed node is not in the graph
        if deployed_node not in G.nodes():
            continue
        
        node_type = G.nodes[deployed_node]['type']
        services_by_node_type[service_type][node_type] += 1
        services_by_type[service_type] += 1
    
    # Create bar chart of services by type
    plt.figure(figsize=(12, 6))
    service_types = list(services_by_type.keys())
    counts = [services_by_type[s] for s in service_types]
    
    plt.bar(service_types, counts, color='#9b59b6')
    plt.title('Services by Type', fontsize=16)
    plt.xlabel('Service Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create stacked bar chart of services by node type
    plt.figure(figsize=(12, 6))
    
    cloud_counts = []
    fog_counts = []
    edge_counts = []
    
    for service_type in service_types:
        cloud_counts.append(services_by_node_type[service_type].get('cloud', 0))
        fog_counts.append(services_by_node_type[service_type].get('fog', 0))
        edge_counts.append(services_by_node_type[service_type].get('edge', 0))
    
    plt.bar(service_types, cloud_counts, label='Cloud', color='#36AE7C')
    plt.bar(service_types, fog_counts, bottom=cloud_counts, label='Fog', color='#F9D923')
    
    # Calculate the bottom for edge counts
    bottom_edge = [c + f for c, f in zip(cloud_counts, fog_counts)]
    plt.bar(service_types, edge_counts, bottom=bottom_edge, label='Edge', color='#EB5353')
    
    plt.title('Services Deployment by Node Type', fontsize=16)
    plt.xlabel('Service Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file.replace('.png', '_by_node_type.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Services distribution visualizations saved to {output_file} and {output_file.replace('.png', '_by_node_type.png')}")

def plot_application_workflows(output_file_prefix="application_workflow"):
    """Create visualizations of application workflows"""
    # Media Processing
    media_G = nx.DiGraph()
    media_nodes = [
        ('fDocAnalysis', {'software': 'py3', 'memory': 1024, 'vcpu': 2}),
        ('fProcDoc', {'software': 'js', 'memory': 1024, 'vcpu': 1}),
        ('fPayAppr', {'software': 'py3', 'memory': 256, 'vcpu': 2}),
        ('fNotify', {'software': 'py3', 'memory': 128, 'vcpu': 2}),
        ('fArchive', {'software': 'py3', 'memory': 256, 'vcpu': 2})
    ]
    media_G.add_nodes_from(media_nodes)
    media_edges = [
        ('fDocAnalysis', 'fProcDoc', {'label': 'seq'}),
        ('fProcDoc', 'fPayAppr', {'label': 'if-condition'}),
        ('fPayAppr', 'fNotify', {'label': 'then'}),
        ('fPayAppr', 'fArchive', {'label': 'else'})
    ]
    media_G.add_edges_from(media_edges)
    
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(media_G, seed=42)
    nx.draw_networkx_nodes(media_G, pos, node_color='#9b59b6', alpha=0.8)
    nx.draw_networkx_labels(media_G, pos)
    nx.draw_networkx_edges(media_G, pos, edge_color='gray', arrows=True)
    edge_labels = {(u, v): d['label'] for u, v, d in media_G.edges(data=True)}
    nx.draw_networkx_edge_labels(media_G, pos, edge_labels=edge_labels)
    
    plt.title('Media Processing Application Workflow')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_file_prefix}_media.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Siotec2022
    siotec_G = nx.DiGraph()
    siotec_nodes = [
        ('fLogin', {'software': 'js', 'memory': 1024, 'vcpu': 2}),
        ('fCrop', {'software': 'py3, numPy', 'memory': 2048, 'vcpu': 4}),
        ('fGeo', {'software': 'js', 'memory': 256, 'vcpu': 2}),
        ('fDCC', {'software': 'js', 'memory': 128, 'vcpu': 2}),
        ('fCheckDCC', {'software': 'js', 'memory': 128, 'vcpu': 2})
    ]
    siotec_G.add_nodes_from(siotec_nodes)
    siotec_edges = [
        ('fLogin', 'fCrop', {'label': 'seq'}),
        ('fLogin', 'fDCC', {'label': 'seq'}),
        ('fDCC', 'fCheckDCC', {'label': 'then'}),
        ('fCrop', 'fGeo', {'label': 'seq'})
    ]
    siotec_G.add_edges_from(siotec_edges)
    
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(siotec_G, seed=42)
    nx.draw_networkx_nodes(siotec_G, pos, node_color='#e74c3c', alpha=0.8)
    nx.draw_networkx_labels(siotec_G, pos)
    nx.draw_networkx_edges(siotec_G, pos, edge_color='gray', arrows=True)
    edge_labels = {(u, v): d['label'] for u, v, d in siotec_G.edges(data=True)}
    nx.draw_networkx_edge_labels(siotec_G, pos, edge_labels=edge_labels)
    
    plt.title('Siotec2022 Application Workflow')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_file_prefix}_siotec.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Stock Market
    stock_G = nx.DiGraph()
    stock_nodes = [
        ('fCheck', {'software': 'py3', 'memory': 512, 'vcpu': 2}),
        ('fBuyOrSell', {'software': 'py3', 'memory': 2048, 'vcpu': 4}),
        ('fSell', {'software': 'js, py3', 'memory': 256, 'vcpu': 2}),
        ('fBuy', {'software': 'js, py3', 'memory': 256, 'vcpu': 2})
    ]
    stock_G.add_nodes_from(stock_nodes)
    stock_edges = [
        ('fCheck', 'fBuyOrSell', {'label': 'seq'}),
        ('fBuyOrSell', 'fSell', {'label': 'then'}),
        ('fBuyOrSell', 'fBuy', {'label': 'else'})
    ]
    stock_G.add_edges_from(stock_edges)
    
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(stock_G, seed=42)
    nx.draw_networkx_nodes(stock_G, pos, node_color='#2ecc71', alpha=0.8)
    nx.draw_networkx_labels(stock_G, pos)
    nx.draw_networkx_edges(stock_G, pos, edge_color='gray', arrows=True)
    edge_labels = {(u, v): d['label'] for u, v, d in stock_G.edges(data=True)}
    nx.draw_networkx_edge_labels(stock_G, pos, edge_labels=edge_labels)
    
    plt.title('Stock Market Application Workflow')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_file_prefix}_stock.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Application workflow visualizations saved to {output_file_prefix}_*.png")

def plot_node_stats(report_data, output_file="node_stats.png"):
    """Plot node statistics from the simulation report"""
    if not report_data or 'nodes' not in report_data:
        print("No node statistics available in the report")
        return
    
    node_load = report_data['nodes'].get('load', {})
    if not node_load:
        print("No node load data available")
        return
    
    # Extract node types from node IDs
    node_types = {}
    for node_id in node_load.keys():
        if 'cloud' in node_id:
            node_types[node_id] = 'cloud'
        elif 'fog' in node_id:
            node_types[node_id] = 'fog'
        elif 'edge' in node_id:
            node_types[node_id] = 'edge'
        else:
            node_types[node_id] = 'unknown'
    
    # Calculate average load by node type and epoch
    load_by_type = defaultdict(lambda: defaultdict(list))
    
    for node_id, epochs in node_load.items():
        node_type = node_types[node_id]
        for epoch, stats in epochs.items():
            load_by_type[node_type][int(epoch)].append(float(stats['load']))
    
    # Calculate average for each epoch and type
    avg_load = {}
    for node_type in load_by_type:
        avg_load[node_type] = {}
        for epoch, loads in load_by_type[node_type].items():
            avg_load[node_type][epoch] = sum(loads) / len(loads) if loads else 0
    
    # Plot average load over time by node type
    plt.figure(figsize=(15, 8))
    
    for node_type, color in [('cloud', '#36AE7C'), ('fog', '#F9D923'), ('edge', '#EB5353')]:
        if node_type in avg_load:
            epochs = sorted(avg_load[node_type].keys())
            loads = [avg_load[node_type][e] for e in epochs]
            plt.plot(epochs, loads, label=f"{node_type.capitalize()} Nodes", color=color, linewidth=2)
    
    plt.title('Average Node Load by Type Over Time', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Average Load')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a heatmap of the top 20 nodes with highest average load
    node_avg_loads = {}
    for node_id, epochs in node_load.items():
        loads = [float(stats['load']) for stats in epochs.values()]
        node_avg_loads[node_id] = sum(loads) / len(loads) if loads else 0
    
    # Get top 20 nodes
    top_nodes = sorted(node_avg_loads.items(), key=lambda x: x[1], reverse=True)[:20]
    top_node_ids = [node[0] for node in top_nodes]
    
    # Create data for heatmap
    heatmap_data = []
    node_ids = []
    
    for node_id in top_node_ids:
        node_ids.append(node_id)
        epoch_data = {}
        for epoch, stats in node_load[node_id].items():
            epoch_data[int(epoch)] = float(stats['load'])
        heatmap_data.append(epoch_data)
    
    # Convert to DataFrame
    max_epoch = max(int(e) for node in node_load.values() for e in node.keys())
    df_data = []
    
    for i, node_id in enumerate(node_ids):
        for epoch in range(max_epoch + 1):
            if epoch in heatmap_data[i]:
                df_data.append({
                    'Node': node_id,
                    'Epoch': epoch,
                    'Load': heatmap_data[i][epoch]
                })
    
    if df_data:
        df = pd.DataFrame(df_data)
        pivot_df = df.pivot(index='Node', columns='Epoch', values='Load')
        
        # Plot heatmap
        plt.figure(figsize=(18, 10))
        plt.pcolormesh(pivot_df.columns, range(len(pivot_df.index)), pivot_df.values, cmap='viridis', shading='auto')
        plt.colorbar(label='Load')
        plt.yticks(range(len(pivot_df.index)), pivot_df.index)
        plt.title('Node Load Heatmap (Top 20 Nodes)', fontsize=16)
        plt.xlabel('Epoch')
        plt.ylabel('Node')
        plt.tight_layout()
        
        plt.savefig(output_file.replace('.png', '_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Node statistics visualizations saved to {output_file} and {output_file.replace('.png', '_heatmap.png')}")

def plot_infra_events(report_data, output_file="infrastructure_events.png"):
    """Plot infrastructure events (crashes and resurrections) from the report"""
    if not report_data or 'nodes' not in report_data or 'links' not in report_data:
        print("No infrastructure events data available in the report")
        return
    
    node_events = report_data['nodes'].get('events', [])
    link_events = report_data['links'].get('events', [])
    
    if not node_events and not link_events:
        print("No events data available")
        return
    
    # Process node events
    node_crashes_by_epoch = defaultdict(int)
    node_resurrections_by_epoch = defaultdict(int)
    
    for event in node_events:
        epoch = int(event['epoch'])
        if event['type'] == 'crash':
            node_crashes_by_epoch[epoch] += 1
        elif event['type'] == 'resurrection':
            node_resurrections_by_epoch[epoch] += 1
    
    # Process link events
    link_crashes_by_epoch = defaultdict(int)
    link_resurrections_by_epoch = defaultdict(int)
    
    for event in link_events:
        epoch = int(event['epoch'])
        if event['type'] == 'crash':
            link_crashes_by_epoch[epoch] += 1
        elif event['type'] == 'resurrection':
            link_resurrections_by_epoch[epoch] += 1
    
    # Get all epochs
    all_epochs = set()
    all_epochs.update(node_crashes_by_epoch.keys(), node_resurrections_by_epoch.keys(),
                      link_crashes_by_epoch.keys(), link_resurrections_by_epoch.keys())
    epochs = sorted(all_epochs)
    
    if not epochs:
        print("No epochs with events found")
        return
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # Plot node events
    ax1.bar([e-0.2 for e in epochs], 
            [node_crashes_by_epoch.get(e, 0) for e in epochs], 
            width=0.4, color='#e74c3c', label='Node Crashes')
    ax1.bar([e+0.2 for e in epochs], 
            [node_resurrections_by_epoch.get(e, 0) for e in epochs], 
            width=0.4, color='#2ecc71', label='Node Resurrections')
    
    ax1.set_title('Node Events Over Time', fontsize=14)
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot link events
    ax2.bar([e-0.2 for e in epochs], 
            [link_crashes_by_epoch.get(e, 0) for e in epochs], 
            width=0.4, color='#e67e22', label='Link Crashes')
    ax2.bar([e+0.2 for e in epochs], 
            [link_resurrections_by_epoch.get(e, 0) for e in epochs], 
            width=0.4, color='#3498db', label='Link Resurrections')
    
    ax2.set_title('Link Events Over Time', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot cumulative events
    plt.figure(figsize=(15, 8))
    
    # Calculate cumulative events
    epochs_range = range(max(epochs) + 1)
    
    cum_node_crashes = [sum(node_crashes_by_epoch.get(e, 0) for e in range(epoch+1)) for epoch in epochs_range]
    cum_node_resurrections = [sum(node_resurrections_by_epoch.get(e, 0) for e in range(epoch+1)) for epoch in epochs_range]
    cum_link_crashes = [sum(link_crashes_by_epoch.get(e, 0) for e in range(epoch+1)) for epoch in epochs_range]
    cum_link_resurrections = [sum(link_resurrections_by_epoch.get(e, 0) for e in range(epoch+1)) for epoch in epochs_range]
    
    plt.plot(epochs_range, cum_node_crashes, color='#e74c3c', label='Cumulative Node Crashes', linewidth=2)
    plt.plot(epochs_range, cum_node_resurrections, color='#2ecc71', label='Cumulative Node Resurrections', linewidth=2)
    plt.plot(epochs_range, cum_link_crashes, color='#e67e22', label='Cumulative Link Crashes', linewidth=2)
    plt.plot(epochs_range, cum_link_resurrections, color='#3498db', label='Cumulative Link Resurrections', linewidth=2)
    
    plt.title('Cumulative Infrastructure Events Over Time', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Cumulative Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file.replace('.png', '_cumulative.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Infrastructure events visualizations saved to {output_file} and {output_file.replace('.png', '_cumulative.png')}")

def create_html_report(output_dir):
    """Create an HTML report combining all visualizations"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LambdaFogSim Infrastructure Visualization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1, h2 {
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            .section {
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }
            .image-container {
                text-align: center;
                margin: 20px 0;
            }
            img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
            }
            footer {
                margin-top: 30px;
                text-align: center;
                color: #777;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LambdaFogSim Infrastructure Visualization Report</h1>
            
            <div class="section">
                <h2>Infrastructure Network</h2>
                <p>This visualization shows the complete infrastructure topology with cloud, fog, and edge nodes and their connections.</p>
                <div class="image-container">
                    <img src="infrastructure_network.png" alt="Infrastructure Network">
                </div>
            </div>
            
            <div class="section">
                <h2>Security Levels</h2>
                <p>Distribution of security capabilities across the infrastructure.</p>
                <div class="image-container">
                    <img src="security_levels.png" alt="Security Levels">
                </div>
            </div>
            
            <div class="section">
                <h2>Software Capabilities</h2>
                <p>Distribution of software capabilities across the infrastructure.</p>
                <div class="image-container">
                    <img src="software_capabilities.png" alt="Software Capabilities">
                </div>
                <div class="image-container">
                    <img src="software_capabilities_by_node_type.png" alt="Software Capabilities by Node Type">
                </div>
            </div>
            
            <div class="section">
                <h2>Services Distribution</h2>
                <p>Distribution of services across the infrastructure.</p>
                <div class="image-container">
                    <img src="services_distribution.png" alt="Services Distribution">
                </div>
                <div class="image-container">
                    <img src="services_distribution_by_node_type.png" alt="Services Distribution by Node Type">
                </div>
            </div>
            
            <div class="section">
                <h2>Application Workflows</h2>
                <p>Visualization of application workflows.</p>
                <h3>Media Processing</h3>
                <div class="image-container">
                    <img src="application_workflow_media.png" alt="Media Processing Workflow">
                </div>
                <h3>Siotec2022</h3>
                <div class="image-container">
                    <img src="application_workflow_siotec.png" alt="Siotec2022 Workflow">
                </div>
                <h3>Stock Market</h3>
                <div class="image-container">
                    <img src="application_workflow_stock.png" alt="Stock Market Workflow">
                </div>
            </div>
            
            <div class="section">
                <h2>Node Statistics</h2>
                <p>Node load statistics over time.</p>
                <div class="image-container">
                    <img src="node_stats.png" alt="Node Statistics">
                </div>
                <div class="image-container">
                    <img src="node_stats_heatmap.png" alt="Node Load Heatmap">
                </div>
            </div>
            
            <div class="section">
                <h2>Infrastructure Events</h2>
                <p>Node and link failures and resurrections.</p>
                <div class="image-container">
                    <img src="infrastructure_events.png" alt="Infrastructure Events">
                </div>
                <div class="image-container">
                    <img src="infrastructure_events_cumulative.png" alt="Cumulative Infrastructure Events">
                </div>
            </div>
            
            <footer>
                <p>Generated by LambdaFogSim Infrastructure Visualization</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'infrastructure_report.html'), 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {os.path.join(output_dir, 'infrastructure_report.html')}")

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Parse infrastructure file
    nodes, links, services = parse_infrastructure_file(INFRASTRUCTURE_FILE)
    
    if not nodes or not links:
        print("Failed to parse infrastructure file")
        return
    
    # Create the graph
    G = create_infrastructure_graph(nodes, links)
    
    # Create visualizations
    plot_infrastructure(G, os.path.join(OUTPUT_DIR, 'infrastructure_network.png'))
    plot_security_levels(G, os.path.join(OUTPUT_DIR, 'security_levels.png'))
    plot_software_capabilities(G, os.path.join(OUTPUT_DIR, 'software_capabilities.png'))
    plot_services_distribution(G, services, os.path.join(OUTPUT_DIR, 'services_distribution.png'))
    plot_application_workflows(os.path.join(OUTPUT_DIR, 'application_workflow'))
    
    # Load simulation report if available
    report_data = load_report()
    if report_data:
        plot_node_stats(report_data, os.path.join(OUTPUT_DIR, 'node_stats.png'))
        plot_infra_events(report_data, os.path.join(OUTPUT_DIR, 'infrastructure_events.png'))
    
    # Create HTML report
    create_html_report(OUTPUT_DIR)
    
    print(f"All visualizations have been saved to the {OUTPUT_DIR} directory")
    print(f"Open {os.path.join(OUTPUT_DIR, 'infrastructure_report.html')} in a web browser to view the complete report")

if __name__ == "__main__":
    main()