import os
import json
import pandas as pd
from collections import defaultdict
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath("./src"))

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

def export_placement_data(report_data, output_dir="exports"):
    """Export placement data to CSV"""
    if not report_data or 'applications' not in report_data:
        print("No application data found in report")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract placement data for all applications
    placements_data = []
    replacements_data = []
    
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
            placements_data.append(record)
        
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
            replacements_data.append(record)
    
    # Create DataFrames and export to CSV
    placements_df = pd.DataFrame(placements_data)
    replacements_df = pd.DataFrame(replacements_data)
    
    placements_csv_path = os.path.join(output_dir, 'placements.csv')
    replacements_csv_path = os.path.join(output_dir, 'replacements.csv')
    
    placements_df.to_csv(placements_csv_path, index=False)
    replacements_df.to_csv(replacements_csv_path, index=False)
    
    print(f"Exported placement data to {placements_csv_path}")
    print(f"Exported replacement data to {replacements_csv_path}")
    
    return placements_df, replacements_df

def export_node_data(report_data, output_dir="exports"):
    """Export node data to CSV"""
    if not report_data or 'nodes' not in report_data:
        print("No node data found in report")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract node load data
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
    
    # Extract node event data
    node_events_data = []
    
    for event in report_data['nodes']['events']:
        record = {
            'event_type': event['type'],
            'node_id': event['node_id'],
            'epoch': event['epoch']
        }
        node_events_data.append(record)
    
    # Create DataFrames and export to CSV
    node_load_df = pd.DataFrame(node_load_data)
    node_events_df = pd.DataFrame(node_events_data)
    
    # Sort by node_id and epoch for easier analysis
    if not node_load_df.empty:
        node_load_df = node_load_df.sort_values(['node_id', 'epoch'])
    
    if not node_events_df.empty:
        node_events_df = node_events_df.sort_values(['epoch', 'node_id'])
    
    node_load_csv_path = os.path.join(output_dir, 'node_load.csv')
    node_events_csv_path = os.path.join(output_dir, 'node_events.csv')
    
    node_load_df.to_csv(node_load_csv_path, index=False)
    node_events_df.to_csv(node_events_csv_path, index=False)
    
    print(f"Exported node load data to {node_load_csv_path}")
    print(f"Exported node events data to {node_events_csv_path}")
    
    return node_load_df, node_events_df

def export_link_data(report_data, output_dir="exports"):
    """Export link event data to CSV"""
    if not report_data or 'links' not in report_data:
        print("No link data found in report")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract link event data
    link_events_data = []
    
    for event in report_data['links']['events']:
        record = {
            'event_type': event['type'],
            'first_node_id': event['first_node_id'],
            'second_node_id': event['second_node_id'],
            'epoch': event['epoch']
        }
        link_events_data.append(record)
    
    # Create DataFrame and export to CSV
    link_events_df = pd.DataFrame(link_events_data)
    
    # Sort by epoch for easier analysis
    if not link_events_df.empty:
        link_events_df = link_events_df.sort_values('epoch')
    
    link_events_csv_path = os.path.join(output_dir, 'link_events.csv')
    
    link_events_df.to_csv(link_events_csv_path, index=False)
    
    print(f"Exported link events data to {link_events_csv_path}")
    
    return link_events_df

def export_application_summary(report_data, output_dir="exports"):
    """Export application performance summary to CSV"""
    if not report_data or 'applications' not in report_data:
        print("No application data found in report")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract application summary data
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
    
    # Create DataFrame and export to CSV
    app_summary_df = pd.DataFrame(app_summary_data)
    
    app_summary_csv_path = os.path.join(output_dir, 'application_summary.csv')
    
    app_summary_df.to_csv(app_summary_csv_path, index=False)
    
    print(f"Exported application summary data to {app_summary_csv_path}")
    
    return app_summary_df

def export_all_data(report_path=None, output_dir="exports"):
    """Export all simulation data to CSV files"""
    report_data = load_report(report_path)
    
    if not report_data:
        print("No report data to export")
        return
    
    print(f"\nExporting all simulation data to {output_dir}/")
    
    export_placement_data(report_data, output_dir)
    export_node_data(report_data, output_dir)
    export_link_data(report_data, output_dir)
    export_application_summary(report_data, output_dir)
    
    print("\nAll data exported successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export LambdaFogSim simulation data to CSV")
    parser.add_argument("-r", "--report", help="Path to JSON report file (default: latest report)")
    parser.add_argument("-o", "--output", default="exports", help="Output directory for CSV files")
    
    args = parser.parse_args()
    
    export_all_data(args.report, args.output)