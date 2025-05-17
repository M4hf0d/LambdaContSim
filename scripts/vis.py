import os
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
from collections import defaultdict
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.abspath("./src"))

from infrastructure.utils import plot_infrastructure
from infrastructure.logical_infrastructure import LogicalInfrastructure
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend best for saving files
def find_latest_report():
    """Find the most recent report file in the reports directory"""
    reports_dir = os.path.join(os.getcwd(), "reports")
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

def plot_placement_statistics(report_data):
    """Plot placement success rates"""
    if not report_data:
        return
    
    # Extract placement data
    general = report_data['general']
    placements = general['placements']
    replacements = general['replacements']
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Attempt vs Success counts
    categories = ['Placements', 'Replacements', 'Total']
    attempts = [placements['total_attempts'], replacements['total_attempts'], 
                placements['total_attempts'] + replacements['total_attempts']]
    successes = [placements['total_successes'], replacements['total_successes'],
                placements['total_successes'] + replacements['total_successes']]
    failures = [attempts[0]-successes[0], attempts[1]-successes[1], attempts[2]-successes[2]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axs[0].bar(x - width/2, successes, width, label='Successes', color='green')
    axs[0].bar(x + width/2, failures, width, label='Failures', color='red')
    
    axs[0].set_ylabel('Count')
    axs[0].set_title('Placement and Replacement Results')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(categories)
    axs[0].legend()
    
    # Plot 2: Average execution times
    success_times = [placements['avg_success_execution_time'], 
                    replacements['avg_success_execution_time'],
                    general['global']['avg_success_execution_time']]
    
    failure_times = [placements['avg_failure_execution_time'], 
                    replacements['avg_failure_execution_time'],
                    general['global']['avg_failure_execution_time']]
    
    axs[1].bar(x - width/2, success_times, width, label='Success Time', color='green')
    axs[1].bar(x + width/2, failure_times, width, label='Failure Time', color='red')
    
    axs[1].set_ylabel('Seconds')
    axs[1].set_title('Average Execution Times')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(categories)
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig('placement_statistics.png')
    plt.show()
    
    return fig

def plot_application_success_rates(report_data):
    """Plot success rates for each application"""
    if not report_data:
        return
    
    applications = report_data['applications']
    app_names = list(applications.keys())
    
    # Extract placement and replacement stats
    placement_attempts = []
    placement_successes = []
    replacement_attempts = []
    replacement_successes = []
    
    for app in app_names:
        placement_attempts.append(applications[app]['placements']['attempts'])
        placement_successes.append(applications[app]['placements']['successes'])
        replacement_attempts.append(applications[app]['replacements']['attempts'])
        replacement_successes.append(applications[app]['replacements']['successes'])
    
    # Calculate success rates
    placement_rates = [s/a if a > 0 else 0 for s, a in zip(placement_successes, placement_attempts)]
    replacement_rates = [s/a if a > 0 else 0 for s, a in zip(replacement_successes, replacement_attempts)]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(app_names))
    width = 0.35
    
    ax.bar(x - width/2, placement_rates, width, label='Placement Success Rate', color='blue')
    ax.bar(x + width/2, replacement_rates, width, label='Replacement Success Rate', color='orange')
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Success Rate')
    ax.set_title('Application Success Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(app_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('application_success_rates.png')
    plt.show()
    
    return fig

def plot_node_crash_events(report_data):
    """Plot node crash and resurrection events over time"""
    if not report_data or 'nodes' not in report_data:
        return
    
    node_events = report_data['nodes']['events']
    
    # Group events by epoch
    crashes_by_epoch = defaultdict(int)
    resurrections_by_epoch = defaultdict(int)
    
    for event in node_events:
        epoch = event['epoch']
        if event['type'] == 'crash':
            crashes_by_epoch[epoch] += 1
        elif event['type'] == 'resurrection':
            resurrections_by_epoch[epoch] += 1
    
    # Find the maximum epoch
    max_epoch = max(max(crashes_by_epoch.keys(), default=0), max(resurrections_by_epoch.keys(), default=0))
    
    # Create arrays for plotting
    epochs = list(range(max_epoch + 1))
    crashes = [crashes_by_epoch[e] for e in epochs]
    resurrections = [resurrections_by_epoch[e] for e in epochs]
    
    # Create cumulative count arrays
    cum_crashes = np.cumsum(crashes)
    cum_resurrections = np.cumsum(resurrections)
    net_failures = cum_crashes - cum_resurrections
    
    # Create plots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot per-epoch events
    axs[0].bar(epochs, crashes, color='red', alpha=0.7, label='Crashes')
    axs[0].bar(epochs, resurrections, color='green', alpha=0.7, label='Resurrections')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Count')
    axs[0].set_title('Node Crashes and Resurrections per Epoch')
    axs[0].legend()
    
    # Plot cumulative events
    axs[1].plot(epochs, cum_crashes, 'r-', label='Cumulative Crashes')
    axs[1].plot(epochs, cum_resurrections, 'g-', label='Cumulative Resurrections')
    axs[1].plot(epochs, net_failures, 'b-', label='Net Failed Nodes')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Count')
    axs[1].set_title('Cumulative Node Failures Over Time')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig('node_crash_events.png')
    plt.show()
    
    return fig

def plot_link_crash_events(report_data):
    """Plot link crash and resurrection events over time"""
    if not report_data or 'links' not in report_data:
        return
    
    link_events = report_data['links']['events']
    
    # Group events by epoch
    crashes_by_epoch = defaultdict(int)
    resurrections_by_epoch = defaultdict(int)
    
    for event in link_events:
        epoch = event['epoch']
        if event['type'] == 'crash':
            crashes_by_epoch[epoch] += 1
        elif event['type'] == 'resurrection':
            resurrections_by_epoch[epoch] += 1
    
    # Find the maximum epoch
    max_epoch = max(max(crashes_by_epoch.keys(), default=0), max(resurrections_by_epoch.keys(), default=0))
    
    # Create arrays for plotting
    epochs = list(range(max_epoch + 1))
    crashes = [crashes_by_epoch[e] for e in epochs]
    resurrections = [resurrections_by_epoch[e] for e in epochs]
    
    # Create cumulative count arrays
    cum_crashes = np.cumsum(crashes)
    cum_resurrections = np.cumsum(resurrections)
    net_failures = cum_crashes - cum_resurrections
    
    # Create plots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot per-epoch events
    axs[0].bar(epochs, crashes, color='red', alpha=0.7, label='Crashes')
    axs[0].bar(epochs, resurrections, color='green', alpha=0.7, label='Resurrections')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Count')
    axs[0].set_title('Link Crashes and Resurrections per Epoch')
    axs[0].legend()
    
    # Plot cumulative events
    axs[1].plot(epochs, cum_crashes, 'r-', label='Cumulative Crashes')
    axs[1].plot(epochs, cum_resurrections, 'g-', label='Cumulative Resurrections')
    axs[1].plot(epochs, net_failures, 'b-', label='Net Failed Links')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Count')
    axs[1].set_title('Cumulative Link Failures Over Time')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig('link_crash_events.png')
    plt.show()
    
    return fig

def plot_node_load_heatmap(report_data):
    """Create a heatmap of node loads over time"""
    if not report_data or 'nodes' not in report_data:
        return
    
    node_stats = report_data['nodes']['load']
    
    # Extract nodes and epochs
    nodes = list(node_stats.keys())
    all_epochs = set()
    for node, epochs in node_stats.items():
        all_epochs.update(map(int, epochs.keys()))
    
    epochs = sorted(all_epochs)
    
    # Create a matrix of loads
    load_matrix = np.zeros((len(nodes), len(epochs)))
    
    for i, node in enumerate(nodes):
        for j, epoch in enumerate(epochs):
            epoch_str = str(epoch)
            if epoch_str in node_stats[node]:
                load_matrix[i, j] = node_stats[node][epoch_str]['load']
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(load_matrix, aspect='auto', cmap='viridis')
    
    # Add labels
    ax.set_yticks(np.arange(len(nodes)))
    ax.set_yticklabels(nodes)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Node')
    ax.set_title('Node Load Over Time')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Load', rotation=-90, va="bottom")
    
    plt.tight_layout()
    plt.savefig('node_load_heatmap.png')
    plt.show()
    
    return fig

def visualize_final_infrastructure():
    """Visualize the final infrastructure state"""
    try:
        infrastructure_file = "generated_infrastructure.pl"
        if not os.path.exists(infrastructure_file):
            print(f"Infrastructure file not found at {infrastructure_file}")
            return None
            
        infrastructure = LogicalInfrastructure.loads(infrastructure_file)
        if infrastructure:
            print("Visualizing final infrastructure state...")
            plot_infrastructure(infrastructure)
            return True
        else:
            print("Failed to load infrastructure")
            return None
    except Exception as e:
        print(f"Error visualizing infrastructure: {e}")
        return None

def generate_summary_report(report_data):
    """Generate a text summary report of the simulation"""
    if not report_data:
        return None
        
    general = report_data['general']
    
    # Create summary text
    summary = [
        "# LambdaFogSim Simulation Summary Report",
        "",
        f"Total epochs: {general['epochs']}",
        f"Random seed: {general['seed']}",
        "",
        "## Placement Statistics",
        f"- Total placement attempts: {general['placements']['total_attempts']}",
        f"- Successful placements: {general['placements']['total_successes']} ({general['placements']['total_successes']/general['placements']['total_attempts']:.1%} success rate)",
        f"- Average placement time: {general['placements']['avg_total_execution_time']:.3f} seconds",
        "",
        "## Replacement Statistics",
        f"- Total replacement attempts: {general['replacements']['total_attempts']}",
        f"- Successful replacements: {general['replacements']['total_successes']} ({general['replacements']['total_successes']/general['replacements']['total_attempts'] if general['replacements']['total_attempts'] > 0 else 0:.1%} success rate)",
        f"- Average replacement time: {general['replacements']['avg_total_execution_time']:.3f} seconds",
        "",
        "## Application Performance",
    ]
    
    # Add per-application statistics
    for app_name, app_data in report_data['applications'].items():
        placement_success_rate = app_data['placements']['successes'] / app_data['placements']['attempts'] if app_data['placements']['attempts'] > 0 else 0
        replacement_success_rate = app_data['replacements']['successes'] / app_data['replacements']['attempts'] if app_data['replacements']['attempts'] > 0 else 0
        
        summary.extend([
            f"### {app_name}",
            f"- Placement success rate: {placement_success_rate:.1%} ({app_data['placements']['successes']}/{app_data['placements']['attempts']})",
            f"- Replacement success rate: {replacement_success_rate:.1%} ({app_data['replacements']['successes']}/{app_data['replacements']['attempts']})",
            f"- Average placement time: {app_data['placements']['avg_execution_time']:.3f} seconds",
            f"- Average replacement time: {app_data['replacements']['avg_execution_time']:.3f} seconds",
            ""
        ])
    
    # Count infrastructure events
    node_events = report_data['nodes']['events']
    link_events = report_data['links']['events']
    
    node_crashes = sum(1 for event in node_events if event['type'] == 'crash')
    node_resurrections = sum(1 for event in node_events if event['type'] == 'resurrection')
    link_crashes = sum(1 for event in link_events if event['type'] == 'crash')
    link_resurrections = sum(1 for event in link_events if event['type'] == 'resurrection')
    
    summary.extend([
        "## Infrastructure Events",
        f"- Node crashes: {node_crashes}",
        f"- Node resurrections: {node_resurrections}",
        f"- Link crashes: {link_crashes}",
        f"- Link resurrections: {link_resurrections}",
        ""
    ])
    
    # Write to file
    report_text = "\n".join(summary)
    with open('simulation_summary.md', 'w') as f:
        f.write(report_text)
    
    print(f"Summary report saved to simulation_summary.md")
    return report_text

def main():
    # Load the simulation report
    report_data = load_report()
    
    if not report_data:
        print("No report data available. Please run a simulation first.")
        return
    
    # Generate and display visualizations
    print("\n=== Generating Visualizations ===\n")
    
    print("1. Placement Statistics")
    plot_placement_statistics(report_data)
    
    print("\n2. Application Success Rates")
    plot_application_success_rates(report_data)
    
    print("\n3. Node Crash Events")
    plot_node_crash_events(report_data)
    
    print("\n4. Link Crash Events")
    plot_link_crash_events(report_data)
    
    print("\n5. Node Load Heatmap")
    plot_node_load_heatmap(report_data)
    
    print("\n6. Final Infrastructure State")
    visualize_final_infrastructure()
    
    print("\n7. Generating Summary Report")
    generate_summary_report(report_data)
    
    print("\nAll visualizations and reports have been saved.")

if __name__ == "__main__":
    main()