import os
import pandas as pd
import sys
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

def explore_csv_files(directory=None):
    """
    Explore all CSV files in the specified directory and display their columns and head data
    """
    # If no directory specified, use the directory where this script is located
    if directory is None:
        directory = os.path.dirname(os.path.abspath(__file__))
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"{Fore.RED}Error: Directory '{directory}' does not exist{Style.RESET_ALL}")
        return
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"{Fore.YELLOW}No CSV files found in '{directory}'{Style.RESET_ALL}")
        return
    
    print(f"{Fore.GREEN}Found {len(csv_files)} CSV files in '{directory}'{Style.RESET_ALL}\n")
    
    # Rest of your function remains the same...
    # Process each CSV file
    for i, file_name in enumerate(csv_files, 1):
        file_path = os.path.join(directory, file_name)
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Print file information with nice formatting
            print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}File {i}/{len(csv_files)}: {file_name}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            
            # Print basic stats
            print(f"{Fore.YELLOW}Dimensions: {df.shape[0]} rows × {df.shape[1]} columns{Style.RESET_ALL}")
            
            # Print column names and types
            print(f"\n{Fore.GREEN}Columns:{Style.RESET_ALL}")
            for col_name, dtype in zip(df.columns, df.dtypes):
                print(f"  - {Fore.WHITE}{col_name}{Style.RESET_ALL} ({Fore.BLUE}{dtype}{Style.RESET_ALL})")
            
            # Print first 5 rows
            print(f"\n{Fore.GREEN}First 5 rows:{Style.RESET_ALL}")
            pd.set_option('display.max_columns', None)  # Show all columns
            pd.set_option('display.width', 200)  # Set width to avoid wrapping
            print(df.head().to_string())
            
            # Print some basic statistics if numeric columns are present
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                print(f"\n{Fore.GREEN}Numeric column statistics:{Style.RESET_ALL}")
                stats_df = df[numeric_cols].describe().T
                
                # Print only mean, min, max for brevity
                stats_summary = stats_df[['count', 'mean', 'min', 'max']]
                print(stats_summary.to_string())
            
        except Exception as e:
            print(f"{Fore.RED}Error processing file {file_name}: {str(e)}{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}CSV exploration completed.{Style.RESET_ALL}")

def main():
    # Get directory from command line argument or use script directory by default
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    explore_csv_files(directory)

if __name__ == "__main__":
    main()
