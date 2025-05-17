import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import PercentFormatter

# Set style
plt.style.use('ggplot')
sns.set_palette('colorblind')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['figure.dpi'] = 100

# Create output directory
output_dir = "analysis_results"
os.makedirs(output_dir, exist_ok=True)

# Load data from CSV files
def load_dataframes(csv_dir='csv_exports'):
    dfs = {}
    csv_files = ['application_summary.csv', 'link_events.csv', 'node_events.csv', 
                 'node_load.csv', 'placements.csv', 'replacements.csv', 'simulation_overview.csv']
    
    for file in csv_files:
        path = os.path.join(csv_dir, file)
        if os.path.exists(path):
            dfs[file.split('.')[0]] = pd.read_csv(path)
    
    return dfs

# Main analysis function
def analyze_simulation_results(csv_dir='csv_exports'):
    print("Loading data from CSV files...")
    dfs = load_dataframes(csv_dir)
    
    if not dfs:
        print("No data files found!")
        return
    
    print(f"Generating visualizations in '{output_dir}' directory...")
    
    # 1. Application Success Rate Comparison
    plot_app_success_rates(dfs)
    
    # 2. Node Load Distribution Over Time
    plot_node_load_distribution(dfs)
    
    # 3. Node Type Performance Analysis
    plot_node_type_performance(dfs)
    
    # 4. Infrastructure Reliability Analysis
    plot_infrastructure_reliability(dfs)
    
    # 5. Temporal Analysis of Placements and Replacements
    plot_temporal_analysis(dfs)
    
    # 6. Failure Analysis
    plot_failure_analysis(dfs)
    
    # 7. Time Performance Analysis
    plot_time_performance(dfs)
    
    print(f"Analysis complete! Results saved to '{output_dir}' directory.")

def plot_app_success_rates(dfs):
    """1. Compare success rates between different applications"""
    if 'application_summary' not in dfs:
        return
    
    app_summary = dfs['application_summary']
    
    fig, ax = plt.subplots()
    
    x = np.arange(len(app_summary))
    width = 0.35
    
    placement_bars = ax.bar(x - width/2, app_summary['placement_success_rate'], 
                           width, label='Placement Success Rate')
    replacement_bars = ax.bar(x + width/2, app_summary['replacement_success_rate'], 
                             width, label='Replacement Success Rate')
    
    # Add some text for labels, title and custom x-axis tick labels
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rates by Application')
    ax.set_xticks(x)
    ax.set_xticklabels(app_summary['application'])
    ax.legend()
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1%}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(placement_bars)
    autolabel(replacement_bars)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_application_success_rates.png'))
    plt.close()
    
    # Additional chart: Compare attempt vs. success counts
    fig, ax = plt.subplots()
    
    # Plot for placements
    x = np.arange(len(app_summary))
    width = 0.4
    
    attempts_bars = ax.bar(x - width/2, app_summary['placement_attempts'], 
                          width, label='Placement Attempts')
    success_bars = ax.bar(x + width/2, app_summary['placement_successes'], 
                         width, label='Placement Successes')
    
    ax.set_ylabel('Count')
    ax.set_title('Placement Attempts vs. Successes by Application')
    ax.set_xticks(x)
    ax.set_xticklabels(app_summary['application'])
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(attempts_bars)
    autolabel(success_bars)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1b_placement_attempts_vs_successes.png'))
    plt.close()

def plot_node_load_distribution(dfs):
    """2. Analyze node load distribution over time"""
    if 'node_load' not in dfs:
        return
    
    node_load = dfs['node_load']
    
    # Extract node types from node IDs
    node_load['node_type'] = node_load['node_id'].apply(
        lambda x: 'cloud' if 'cloud' in x else ('fog' if 'fog' in x else 'edge')
    )
    
    # Plot 1: Load distribution by node type (boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='node_type', y='load', data=node_load, palette='Set2')
    plt.title('Load Distribution by Node Type')
    plt.xlabel('Node Type')
    plt.ylabel('Load')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2a_node_load_distribution.png'))
    plt.close()
    
    # Plot 2: Load over time for different node types (line plot with confidence interval)
    # Group by epoch and node type, and calculate mean and std
    load_by_epoch = node_load.groupby(['epoch', 'node_type'])['load'].agg(['mean', 'std']).reset_index()
    
    # Plot for each node type
    plt.figure(figsize=(12, 6))
    
    for node_type, color in zip(['cloud', 'fog', 'edge'], ['blue', 'green', 'red']):
        data = load_by_epoch[load_by_epoch['node_type'] == node_type]
        
        # Plot mean with shaded confidence interval
        plt.plot(data['epoch'], data['mean'], label=f'{node_type.capitalize()} Nodes', color=color)
        plt.fill_between(data['epoch'], 
                         data['mean'] - data['std'], 
                         data['mean'] + data['std'], 
                         alpha=0.2, color=color)
    
    plt.title('Average Node Load Over Time by Node Type')
    plt.xlabel('Epoch')
    plt.ylabel('Load')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2b_node_load_over_time.png'))
    plt.close()
    
    # Plot 3: Heatmap of node load over time for top 20 busiest nodes
    # Find the 20 nodes with highest average load
    top_nodes = node_load.groupby('node_id')['load'].mean().nlargest(20).index
    top_node_load = node_load[node_load['node_id'].isin(top_nodes)]
    
    # Create a pivot table for the heatmap
    heatmap_data = top_node_load.pivot_table(
        index='node_id', 
        columns='epoch', 
        values='load',
        aggfunc='mean'
    )
    
    # Select epochs at regular intervals to make the heatmap readable
    max_epochs = heatmap_data.columns.max()
    step = max(1, max_epochs // 50)  # Show at most 50 epochs
    selected_epochs = range(0, max_epochs+1, step)
    heatmap_data = heatmap_data[selected_epochs]
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(heatmap_data, cmap='viridis', yticklabels=1)
    plt.title('Node Load Heatmap (Top 20 Busiest Nodes)')
    plt.xlabel('Epoch')
    plt.ylabel('Node ID')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2c_node_load_heatmap.png'))
    plt.close()

def plot_node_type_performance(dfs):
    """3. Analyze performance across different node types"""
    if 'placements' not in dfs:
        return
    
    placements = dfs['placements']
    
    # Extract node types
    placements['node_type'] = placements['node'].apply(
        lambda x: 'cloud' if 'cloud' in x else ('fog' if 'fog' in x else 'edge')
    )
    
    # Plot 1: Success rate by node type
    node_type_success = placements.groupby('node_type')['success'].agg(['count', 'sum'])
    node_type_success['success_rate'] = node_type_success['sum'] / node_type_success['count']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(node_type_success.index, node_type_success['success_rate'], color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Placement Success Rate by Node Type')
    plt.xlabel('Node Type')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom')
    
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3a_placement_success_by_node_type.png'))
    plt.close()
    
    # Plot 2: Success rate by application and node type
    app_node_success = placements.groupby(['application', 'node_type'])['success'].agg(['count', 'sum']).reset_index()
    app_node_success['success_rate'] = app_node_success['sum'] / app_node_success['count']
    
    # Create a pivot table for easier plotting
    pivot_data = app_node_success.pivot(index='application', columns='node_type', values='success_rate')
    
    # Plot
    ax = pivot_data.plot(kind='bar', figsize=(10, 6), ylim=(0, 1))
    plt.title('Placement Success Rate by Application and Node Type')
    plt.xlabel('Application')
    plt.ylabel('Success Rate')
    plt.legend(title='Node Type')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type='edge', padding=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3b_success_by_app_and_node.png'))
    plt.close()
    
    # Plot 3: Average duration by node type
    duration_by_node = placements.groupby('node_type')['duration'].mean().reset_index()
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(duration_by_node['node_type'], duration_by_node['duration'], color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Average Placement Duration by Node Type')
    plt.xlabel('Node Type')
    plt.ylabel('Duration (ms)')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3c_placement_duration_by_node.png'))
    plt.close()

def plot_infrastructure_reliability(dfs):
    """4. Analyze infrastructure reliability over time"""
    if 'node_events' not in dfs or 'link_events' not in dfs:
        return
    
    node_events = dfs['node_events']
    link_events = dfs['link_events']
    
    # Prepare data for node failure/recovery over time
    node_failures = node_events[node_events['type'] == 'crash'].groupby('epoch').size()
    node_recoveries = node_events[node_events['type'] == 'resurrection'].groupby('epoch').size()
    
    # Calculate cumulative failures and recoveries
    epochs = range(int(max(node_events['epoch'].max(), link_events['epoch'].max())) + 1)
    
    cum_node_failures = np.zeros(len(epochs))
    cum_node_recoveries = np.zeros(len(epochs))
    
    for epoch, count in node_failures.items():
        epoch_idx = int(epoch)
        if epoch_idx < len(epochs):
            cum_node_failures[epoch_idx:] += count
            
    for epoch, count in node_recoveries.items():
        epoch_idx = int(epoch)
        if epoch_idx < len(epochs):
            cum_node_recoveries[epoch_idx:] += count
    
    # Net failed nodes over time
    net_failed_nodes = cum_node_failures - cum_node_recoveries
    
    # Similar calculations for links
    link_failures = link_events[link_events['type'] == 'crash'].groupby('epoch').size()
    link_recoveries = link_events[link_events['type'] == 'resurrection'].groupby('epoch').size()
    
    cum_link_failures = np.zeros(len(epochs))
    cum_link_recoveries = np.zeros(len(epochs))
    
    for epoch, count in link_failures.items():
        epoch_idx = int(epoch)
        if epoch_idx < len(epochs):
            cum_link_failures[epoch_idx:] += count
            
    for epoch, count in link_recoveries.items():
        epoch_idx = int(epoch)
        if epoch_idx < len(epochs):
            cum_link_recoveries[epoch_idx:] += count
    
    # Net failed links over time
    net_failed_links = cum_link_failures - cum_link_recoveries
    
    # Plot 1: Nodes and links failures over time
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, net_failed_nodes, 'r-', label='Net Failed Nodes')
    plt.plot(epochs, net_failed_links, 'b-', label='Net Failed Links')
    plt.title('Infrastructure Failures Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4a_infrastructure_failures.png'))
    plt.close()
    
    # Plot 2: Node failures by type
    if len(node_events) > 0:
        # Extract node types
        node_events['node_type'] = node_events['node_id'].apply(
            lambda x: 'cloud' if 'cloud' in x else ('fog' if 'fog' in x else 'edge')
        )
        
        # Count crashes by node type
        crashes_by_type = node_events[node_events['type'] == 'crash'].groupby('node_type').size()
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(crashes_by_type.index, crashes_by_type.values, color=['skyblue', 'lightgreen', 'salmon'])
        plt.title('Node Crashes by Type')
        plt.xlabel('Node Type')
        plt.ylabel('Count')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '4b_node_crashes_by_type.png'))
        plt.close()
    
    # Plot 3: Failure events histogram
    plt.figure(figsize=(12, 6))
    
    bin_count = min(50, len(epochs) // 10)
    
    plt.hist(node_events[node_events['type'] == 'crash']['epoch'], 
             bins=bin_count, alpha=0.5, label='Node Crashes')
    plt.hist(link_events[link_events['type'] == 'crash']['epoch'], 
             bins=bin_count, alpha=0.5, label='Link Crashes')
    
    plt.title('Distribution of Failure Events Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4c_failure_distribution.png'))
    plt.close()

def plot_temporal_analysis(dfs):
    """5. Analyze how placements and replacements vary over time"""
    if 'placements' not in dfs:
        return
    
    placements = dfs['placements']
    
    # Group by epoch range (bins) for better visualization
    max_epoch = placements['epoch'].max()
    bin_count = 20
    bin_size = max_epoch / bin_count
    
    placements['epoch_bin'] = (placements['epoch'] // bin_size).astype(int) * bin_size
    
    # Calculate success rate per epoch bin
    success_over_time = placements.groupby('epoch_bin')['success'].agg(['count', 'sum'])
    success_over_time['success_rate'] = success_over_time['sum'] / success_over_time['count']
    
    # Plot 1: Success rate over time
    plt.figure(figsize=(12, 6))
    plt.plot(success_over_time.index, success_over_time['success_rate'], 'o-', markersize=8)
    plt.title('Placement Success Rate Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5a_success_rate_over_time.png'))
    plt.close()
    
    # Plot 2: Placement attempts and successes over time
    plt.figure(figsize=(12, 6))
    width = bin_size * 0.35
    plt.bar(success_over_time.index - width/2, success_over_time['count'], width=width, 
            label='Attempts', alpha=0.6)
    plt.bar(success_over_time.index + width/2, success_over_time['sum'], width=width, 
            label='Successes', alpha=0.6)
    plt.title('Placement Attempts and Successes Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5b_placement_count_over_time.png'))
    plt.close()
    
    # Plot 3: Duration over time
    duration_by_epoch = placements.groupby('epoch_bin')['duration'].mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(duration_by_epoch.index, duration_by_epoch.values, 'o-', markersize=8)
    plt.title('Average Placement Duration Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Duration (ms)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5c_duration_over_time.png'))
    plt.close()

def plot_failure_analysis(dfs):
    """6. Analyze failures in detail"""
    if 'replacements' not in dfs:
        return
    
    replacements = dfs['replacements']
    
    # Plot 1: Replacement success by crashed node count
    if 'crashed_nodes' in replacements.columns:
        # Count number of crashed nodes in each replacement
        replacements['crashed_node_count'] = replacements['crashed_nodes'].fillna('').apply(
            lambda x: len(x.split(',')) if x else 0
        )
        
        # Group by crashed node count
        success_by_node_count = replacements.groupby('crashed_node_count')['success'].agg(['count', 'sum', 'mean'])
        
        # Only plot if we have meaningful data
        if len(success_by_node_count) > 1:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(success_by_node_count.index, success_by_node_count['mean'], color='skyblue')
            plt.title('Replacement Success Rate by Number of Crashed Nodes')
            plt.xlabel('Number of Crashed Nodes')
            plt.ylabel('Success Rate')
            plt.ylim(0, 1)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.1%}', ha='center', va='bottom')
            
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '6a_replacement_by_crashed_nodes.png'))
            plt.close()
    
    # Plot 2: Replacement duration vs success
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='success', y='duration', data=replacements)
    plt.title('Replacement Duration by Outcome')
    plt.xlabel('Success')
    plt.ylabel('Duration (ms)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6b_replacement_duration_by_outcome.png'))
    plt.close()
    
    # Plot 3: Replacement success rates by application
    app_replacement = replacements.groupby('application')['success'].agg(['count', 'sum', 'mean'])
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(app_replacement.index, app_replacement['mean'], color='lightgreen')
    plt.title('Replacement Success Rate by Application')
    plt.xlabel('Application')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom')
    
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6c_replacement_success_by_app.png'))
    plt.close()

def plot_time_performance(dfs):
    """7. Analyze time performance metrics"""
    if 'application_summary' not in dfs or 'simulation_overview' not in dfs:
        return
    
    app_summary = dfs['application_summary']
    overview = dfs['simulation_overview']
    
    # Plot 1: Placement vs replacement execution time by application
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(app_summary))
    width = 0.35
    
    placement_bars = ax.bar(x - width/2, app_summary['avg_placement_time'], 
                           width, label='Avg Placement Time')
    replacement_bars = ax.bar(x + width/2, app_summary['avg_replacement_time'], 
                             width, label='Avg Replacement Time')
    
    ax.set_ylabel('Time (ms)')
    ax.set_title('Average Execution Times by Application')
    ax.set_xticks(x)
    ax.set_xticklabels(app_summary['application'])
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(placement_bars)
    autolabel(replacement_bars)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7a_execution_times_by_app.png'))
    plt.close()
    
    # Plot 2: Relationship between execution time and success rate for placements
    if 'placements' in dfs:
        placements = dfs['placements']
        
        # Group by application and calculate mean duration and success rate
        app_duration = placements.groupby('application').agg({
            'duration': 'mean',
            'success': ['mean', 'count']
        })
        
        app_duration.columns = ['avg_duration', 'success_rate', 'count']
        
        # Create scatter plot with size proportional to count
        plt.figure(figsize=(10, 6))
        plt.scatter(app_duration['avg_duration'], app_duration['success_rate'], 
                   s=app_duration['count']/5, alpha=0.7)
        
        # Add labels for each point
        for i, app in enumerate(app_duration.index):
            plt.annotate(app, 
                        (app_duration['avg_duration'].iloc[i], app_duration['success_rate'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('Relationship Between Duration and Success Rate')
        plt.xlabel('Average Duration (ms)')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '7b_duration_vs_success.png'))
        plt.close()

    # Plot 3: Execution time distribution for successful vs failed attempts
    if 'placements' in dfs:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=placements, x='duration', hue='success', 
                    element='step', stat='density', common_norm=False)
        plt.title('Execution Time Distribution by Outcome')
        plt.xlabel('Duration (ms)')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '7c_duration_distribution.png'))
        plt.close()

if __name__ == "__main__":
    analyze_simulation_results()
    print(f"Open the '{output_dir}' directory to view the generated visualizations.")