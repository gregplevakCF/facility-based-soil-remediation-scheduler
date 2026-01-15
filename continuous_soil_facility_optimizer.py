"""
Continuous Soil Remediation Facility Optimizer - Streamlit App
Determines optimal treatment cell configuration for continuous daily soil volumes
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import math
import numpy as np

# ============================================================================
# Configuration and Constants
# ============================================================================

# Typical soil density: ~1.5 tons/CY (compacted)
# This can be adjusted based on soil type
SOIL_DENSITY_LB_PER_CY = 2700  # ~1.35 tons/CY

# ============================================================================
# Helper Functions
# ============================================================================

def calculate_workdays_in_cycle(start_date, duration_days, saturday_work, sunday_work):
    """Calculate actual working days considering weekends"""
    workdays = 0
    current = start_date
    
    for _ in range(duration_days):
        day_of_week = current.weekday()
        
        # Weekdays always count
        if day_of_week < 5:
            workdays += 1
        # Saturday
        elif day_of_week == 5 and saturday_work:
            workdays += 1
        # Sunday
        elif day_of_week == 6 and sunday_work:
            workdays += 1
        
        current += timedelta(days=1)
    
    return workdays


def calculate_calendar_days_for_workdays(target_workdays, start_date, saturday_work, sunday_work):
    """Calculate how many calendar days are needed to achieve target workdays"""
    workdays = 0
    calendar_days = 0
    current = start_date
    
    while workdays < target_workdays:
        day_of_week = current.weekday()
        
        # Weekdays always count
        if day_of_week < 5:
            workdays += 1
        # Saturday
        elif day_of_week == 5 and saturday_work:
            workdays += 1
        # Sunday
        elif day_of_week == 6 and sunday_work:
            workdays += 1
        
        calendar_days += 1
        current += timedelta(days=1)
    
    return calendar_days


def calculate_cell_dimensions(cell_volume_cy, depth_ft, aspect_ratio=2.0):
    """Calculate cell length and width from volume and depth
    Uses rectangular cells with consistent aspect ratio (Length:Width)
    Default aspect ratio is 2:1 (length is 2x width)
    """
    # Convert CY to cubic feet
    volume_cf = cell_volume_cy * 27
    
    # For a rectangular cell: Volume = L * W * D
    # With aspect ratio: L = aspect_ratio * W
    # Therefore: Volume = (aspect_ratio * W) * W * D = aspect_ratio * W^2 * D
    # Solving for W: W = sqrt(Volume / (aspect_ratio * D))
    width_ft = math.sqrt(volume_cf / (aspect_ratio * depth_ft))
    length_ft = aspect_ratio * width_ft
    
    return {
        'Length_ft': length_ft,
        'Width_ft': width_ft,
        'Depth_ft': depth_ft,
        'Volume_CY': cell_volume_cy,
        'Volume_CF': volume_cf,
        'Area_SF': length_ft * width_ft,
        'Aspect_Ratio': aspect_ratio
    }


def calculate_loading_time(cell_volume_cy, daily_load_capacity, saturday_work, sunday_work):
    """Calculate how long it takes to load a cell"""
    # Calculate workdays needed to load the cell
    workdays_needed = math.ceil(cell_volume_cy / daily_load_capacity)
    
    # Convert to calendar days considering weekend schedule
    calendar_days = calculate_calendar_days_for_workdays(
        workdays_needed,
        datetime.now(),  # Arbitrary start for calculation
        saturday_work,
        sunday_work
    )
    
    return {
        'workdays': workdays_needed,
        'calendar_days': calendar_days
    }


def calculate_total_cycle_time(cell_volume_cy, daily_load_capacity, daily_unload_capacity,
                               rip_days, treat_days, dry_days,
                               load_saturday, load_sunday,
                               rip_saturday, rip_sunday,
                               treat_saturday, treat_sunday,
                               dry_saturday, dry_sunday,
                               unload_saturday, unload_sunday):
    """Calculate total time for a complete cell cycle"""
    
    # Loading time
    load_workdays = math.ceil(cell_volume_cy / daily_load_capacity)
    load_calendar_days = calculate_calendar_days_for_workdays(
        load_workdays, datetime.now(), load_saturday, load_sunday
    )
    
    # Rip time
    rip_calendar_days = calculate_calendar_days_for_workdays(
        rip_days, datetime.now(), rip_saturday, rip_sunday
    )
    
    # Treatment time
    treat_calendar_days = calculate_calendar_days_for_workdays(
        treat_days, datetime.now(), treat_saturday, treat_sunday
    )
    
    # Drying time
    dry_calendar_days = calculate_calendar_days_for_workdays(
        dry_days, datetime.now(), dry_saturday, dry_sunday
    )
    
    # Unloading time
    unload_workdays = math.ceil(cell_volume_cy / daily_unload_capacity)
    unload_calendar_days = calculate_calendar_days_for_workdays(
        unload_workdays, datetime.now(), unload_saturday, unload_sunday
    )
    
    total_calendar_days = (load_calendar_days + rip_calendar_days + 
                          treat_calendar_days + dry_calendar_days + 
                          unload_calendar_days)
    
    return {
        'load_calendar_days': load_calendar_days,
        'load_workdays': load_workdays,
        'rip_calendar_days': rip_calendar_days,
        'treat_calendar_days': treat_calendar_days,
        'dry_calendar_days': dry_calendar_days,
        'unload_calendar_days': unload_calendar_days,
        'unload_workdays': unload_workdays,
        'total_calendar_days': total_calendar_days
    }


def calculate_cells_needed(daily_volume, cell_volume, cycle_days, buffer_factor=1.1):
    """Calculate minimum number of cells needed for continuous operation
    
    Args:
        daily_volume: CY/day arriving at facility
        cell_volume: CY per cell
        cycle_days: Total calendar days for complete cell cycle
        buffer_factor: Safety factor (default 1.1 = 10% buffer)
    """
    # Volume accumulation during one complete cycle
    volume_per_cycle = daily_volume * cycle_days
    
    # Minimum cells needed (theoretical)
    min_cells_theoretical = volume_per_cycle / cell_volume
    
    # Add buffer and round up
    min_cells_with_buffer = math.ceil(min_cells_theoretical * buffer_factor)
    
    return {
        'min_cells_theoretical': min_cells_theoretical,
        'min_cells_with_buffer': min_cells_with_buffer,
        'volume_per_cycle': volume_per_cycle
    }


def optimize_cell_configuration(daily_volume_cy, daily_load_capacity,
                                daily_unload_capacity, phase_params, weekend_params,
                                max_loading_days=14, min_cell_volume=100, 
                                max_cell_volume=5000, step_size=50):
    """Find all viable cell configurations"""
    
    results = []
    
    # Test different cell sizes
    for cell_volume in range(min_cell_volume, max_cell_volume + 1, step_size):
        
        # Calculate cycle time
        cycle_info = calculate_total_cycle_time(
            cell_volume,
            daily_load_capacity,
            daily_unload_capacity,
            phase_params['rip_days'],
            phase_params['treat_days'],
            phase_params['dry_days'],
            weekend_params['load_saturday'],
            weekend_params['load_sunday'],
            weekend_params['rip_saturday'],
            weekend_params['rip_sunday'],
            weekend_params['treat_saturday'],
            weekend_params['treat_sunday'],
            weekend_params['dry_saturday'],
            weekend_params['dry_sunday'],
            weekend_params['unload_saturday'],
            weekend_params['unload_sunday']
        )
        
        # Skip if loading takes too long
        if cycle_info['load_calendar_days'] > max_loading_days:
            continue
        
        # Calculate cells needed
        cells_info = calculate_cells_needed(
            daily_volume_cy,
            cell_volume,
            cycle_info['total_calendar_days']
        )
        
        # Calculate capacity metrics
        total_facility_capacity = cell_volume * cells_info['min_cells_with_buffer']
        days_of_capacity = total_facility_capacity / daily_volume_cy
        
        # Calculate utilization
        # How much soil can we process per day on average?
        daily_throughput = cell_volume / cycle_info['total_calendar_days']
        utilization = daily_volume_cy / (daily_throughput * cells_info['min_cells_with_buffer'])
        
        # Calculate equipment utilization
        # With staggered cells, how often will loaders be working?
        num_cells = cells_info['min_cells_with_buffer']
        
        # Best case: cells start at regular intervals = cycle_time / num_cells
        cell_start_interval = cycle_info['total_calendar_days'] / num_cells
        
        # If we're loading every X days, and loading takes Y days, equipment utilization is Y/X
        # But we want continuous daily work, so ideal is when loading_days >= start_interval
        if cycle_info['load_workdays'] >= cell_start_interval:
            # Overlapping loading periods - good for equipment utilization
            equipment_utilization = min(1.0, cycle_info['load_workdays'] / max(1, cell_start_interval))
        else:
            # Gaps between loading periods - equipment sits idle
            equipment_utilization = cycle_info['load_workdays'] / max(1, cell_start_interval)
        
        # Calculate daily work consistency
        # Ideal: soil loaded per day ≈ daily incoming volume
        avg_daily_load = cell_volume / cycle_info['load_calendar_days']
        load_consistency = min(1.0, avg_daily_load / daily_volume_cy) if daily_volume_cy > 0 else 0
        
        # Calculate unload equipment utilization
        if cycle_info['unload_workdays'] >= cell_start_interval:
            unload_equipment_utilization = min(1.0, cycle_info['unload_workdays'] / max(1, cell_start_interval))
        else:
            unload_equipment_utilization = cycle_info['unload_workdays'] / max(1, cell_start_interval)
        
        # Combined equipment utilization (average of load and unload)
        combined_equipment_util = (equipment_utilization + unload_equipment_utilization) / 2
        
        # Score: prefer fewer cells, good utilization, AND high equipment utilization
        # Lower score is better
        # Heavily penalize poor equipment utilization (idle equipment is expensive)
        score = (
            cells_info['min_cells_with_buffer'] * 100 +  # Capital cost
            (1 - utilization) * 500 +  # Facility utilization
            (1 - combined_equipment_util) * 1500 +  # Equipment utilization (HEAVY WEIGHT)
            (1 - load_consistency) * 300  # Daily work consistency
        )
        
        results.append({
            'cell_volume_cy': cell_volume,
            'num_cells': cells_info['min_cells_with_buffer'],
            'load_days': cycle_info['load_calendar_days'],
            'cycle_days': cycle_info['total_calendar_days'],
            'total_capacity_cy': total_facility_capacity,
            'days_of_capacity': days_of_capacity,
            'utilization': utilization,
            'daily_throughput': daily_throughput * cells_info['min_cells_with_buffer'],
            'equipment_utilization': combined_equipment_util,
            'load_equipment_util': equipment_utilization,
            'unload_equipment_util': unload_equipment_utilization,
            'load_consistency': load_consistency,
            'cell_start_interval': cell_start_interval,
            'score': score,
            'cycle_breakdown': cycle_info
        })
    
    if not results:
        return None
    
    # Sort by score (lower is better)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score').reset_index(drop=True)
    
    return results_df


def is_valid_work_day(date, phase_name, weekend_params):
    """Check if a date is valid for work based on weekend rules"""
    day_of_week = date.weekday()  # Monday=0, Sunday=6
    
    # Weekdays (Mon-Fri) are always valid
    if day_of_week < 5:
        return True
    
    # Saturday
    if day_of_week == 5:
        return weekend_params.get(f'{phase_name.lower()}_saturday', False)
    
    # Sunday
    if day_of_week == 6:
        return weekend_params.get(f'{phase_name.lower()}_sunday', False)
    
    return False


def simulate_facility_schedule(optimal_config, daily_volume_cy, daily_load_capacity, 
                               daily_unload_capacity, phase_params, weekend_params,
                               start_date, simulation_days=180):
    """Simulate facility operations and create detailed schedule"""
    
    cell_volume = optimal_config['cell_volume_cy']
    num_cells = int(optimal_config['num_cells'])
    
    # Initialize cell tracking
    class CellState:
        def __init__(self, cell_num):
            self.cell_num = cell_num
            self.phase = 'Empty'
            self.soil_volume = 0
            self.phase_start_date = None
            self.phase_workdays_completed = 0
            self.flip_num = 0
            
    cells = {i: CellState(i) for i in range(1, num_cells + 1)}
    
    # Track daily activities
    schedule = []
    current_date = start_date
    
    # Accumulator for incoming soil not yet loaded
    soil_waiting = 0
    total_soil_loaded = 0
    total_soil_unloaded = 0
    next_flip_num = 1
    
    for day in range(simulation_days):
        day_activities = {
            'Date': current_date,
            'DayName': current_date.strftime('%A'),
        }
        
        # Add daily incoming volume to waiting pile
        soil_waiting += daily_volume_cy
        
        # Phase 1: Unloading (priority)
        if is_valid_work_day(current_date, 'unload', weekend_params):
            remaining_unload_capacity = daily_unload_capacity
            
            for cell in sorted(cells.values(), key=lambda c: (c.phase != 'ReadyToUnload', c.flip_num)):
                if cell.phase == 'ReadyToUnload' and remaining_unload_capacity > 0:
                    unload_amount = min(cell.soil_volume, remaining_unload_capacity)
                    cell.soil_volume -= unload_amount
                    remaining_unload_capacity -= unload_amount
                    total_soil_unloaded += unload_amount
                    
                    if cell.soil_volume <= 0:
                        cell.phase = 'Empty'
                        cell.flip_num = 0
        
        # Phase 2: Loading
        if is_valid_work_day(current_date, 'load', weekend_params):
            remaining_load_capacity = daily_load_capacity
            
            # Start new cells if empty and soil is waiting
            for cell in sorted(cells.values(), key=lambda c: c.cell_num):
                if cell.phase == 'Empty' and soil_waiting > 0 and remaining_load_capacity > 0:
                    cell.phase = 'Loading'
                    cell.flip_num = next_flip_num
                    next_flip_num += 1
                    cell.phase_start_date = current_date
                    cell.phase_workdays_completed = 0
            
            # Load cells that are in Loading phase
            for cell in sorted(cells.values(), key=lambda c: (c.phase != 'Loading', c.flip_num)):
                if cell.phase == 'Loading' and remaining_load_capacity > 0 and soil_waiting > 0:
                    space_in_cell = cell_volume - cell.soil_volume
                    load_amount = min(space_in_cell, remaining_load_capacity, soil_waiting)
                    
                    cell.soil_volume += load_amount
                    soil_waiting -= load_amount
                    remaining_load_capacity -= load_amount
                    total_soil_loaded += load_amount
                    
                    # Check if loading complete
                    if cell.soil_volume >= cell_volume:
                        cell.phase = 'Rip'
                        cell.phase_start_date = current_date
                        cell.phase_workdays_completed = 0
        
        # Phase 3: Progress other phases
        for cell in cells.values():
            if cell.phase == 'Rip':
                if is_valid_work_day(current_date, 'rip', weekend_params):
                    cell.phase_workdays_completed += 1
                    if cell.phase_workdays_completed >= phase_params['rip_days']:
                        cell.phase = 'Treat'
                        cell.phase_start_date = current_date
                        cell.phase_workdays_completed = 0
                        
            elif cell.phase == 'Treat':
                if is_valid_work_day(current_date, 'treat', weekend_params):
                    cell.phase_workdays_completed += 1
                    if cell.phase_workdays_completed >= phase_params['treat_days']:
                        cell.phase = 'Dry'
                        cell.phase_start_date = current_date
                        cell.phase_workdays_completed = 0
                        
            elif cell.phase == 'Dry':
                if is_valid_work_day(current_date, 'dry', weekend_params):
                    cell.phase_workdays_completed += 1
                    if cell.phase_workdays_completed >= phase_params['dry_days']:
                        cell.phase = 'ReadyToUnload'
                        cell.phase_start_date = current_date
        
        # Record cell states
        for cell_num in range(1, num_cells + 1):
            cell = cells[cell_num]
            phase_display = cell.phase
            if cell.phase not in ['Empty', 'ReadyToUnload']:
                phase_display = f"{cell.phase} ({cell.flip_num})"
            day_activities[f'Cell_{cell_num}_Phase'] = phase_display
            day_activities[f'Cell_{cell_num}_Volume'] = cell.soil_volume
        
        day_activities['SoilWaiting'] = soil_waiting
        day_activities['CumSoilIn'] = total_soil_loaded
        day_activities['CumSoilOut'] = total_soil_unloaded
        
        schedule.append(day_activities)
        current_date += timedelta(days=1)
    
    return pd.DataFrame(schedule)


# ============================================================================
# Main Application
# ============================================================================

def main():
    st.set_page_config(
        page_title="Continuous Soil Facility Optimizer",
        page_icon="🏭",
        layout="wide"
    )
    
    st.title("🏭 Continuous Soil Remediation Facility Optimizer")
    st.markdown("""
    This tool calculates the optimal treatment cell configuration for a continuous-flow 
    soil remediation facility based on daily incoming volume.
    """)
    
    # Initialize session state
    if 'optimization_run' not in st.session_state:
        st.session_state.optimization_run = False
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'selected_config_index' not in st.session_state:
        st.session_state.selected_config_index = None
    if 'parameters' not in st.session_state:
        st.session_state.parameters = {}
    if 'schedule_generated' not in st.session_state:
        st.session_state.schedule_generated = False
    if 'schedule_df' not in st.session_state:
        st.session_state.schedule_df = None
    if 'schedule_start' not in st.session_state:
        st.session_state.schedule_start = None
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("⚙️ Facility Parameters")
        
        # Daily Volume
        st.subheader("Incoming Volume")
        daily_volume = st.number_input(
            "Daily Soil Volume (CY/day)",
            min_value=10,
            max_value=5000,
            value=200,
            step=10,
            help="Average daily volume of soil arriving at facility"
        )
        
        # Equipment Capacity
        st.subheader("Equipment Capacity")
        daily_load_capacity = st.number_input(
            "Daily Loading Capacity (CY/day)",
            min_value=50,
            max_value=2000,
            value=750,
            step=25,
            help="Maximum soil that can be loaded per day"
        )
        
        daily_unload_capacity = st.number_input(
            "Daily Unloading Capacity (CY/day)",
            min_value=50,
            max_value=2000,
            value=750,
            step=25,
            help="Maximum soil that can be unloaded per day"
        )
        
        # Phase Durations
        st.subheader("Treatment Phase Durations (Work Days)")
        rip_days = st.number_input("Rip Duration (days)", min_value=1, value=1, step=1)
        treat_days = st.number_input("Treatment Duration (days)", min_value=1, value=3, step=1)
        dry_days = st.number_input("Drying Duration (days)", min_value=1, value=5, step=1)
        
        # Weekend Working
        st.subheader("Weekend Operations")
        
        st.markdown("**Loading**")
        load_saturday = st.checkbox("Work Saturdays (Load)", value=False)
        load_sunday = st.checkbox("Work Sundays (Load)", value=False)
        
        st.markdown("**Rip**")
        rip_saturday = st.checkbox("Work Saturdays (Rip)", value=True)
        rip_sunday = st.checkbox("Work Sundays (Rip)", value=True)
        
        st.markdown("**Treatment**")
        treat_saturday = st.checkbox("Work Saturdays (Treat)", value=True, 
                                    help="Treatment often continues 7 days/week")
        treat_sunday = st.checkbox("Work Sundays (Treat)", value=True,
                                   help="Treatment often continues 7 days/week")
        
        st.markdown("**Drying**")
        dry_saturday = st.checkbox("Work Saturdays (Dry)", value=True,
                                  help="Drying is passive, counts all days")
        dry_sunday = st.checkbox("Work Sundays (Dry)", value=True,
                                help="Drying is passive, counts all days")
        
        st.markdown("**Unloading**")
        unload_saturday = st.checkbox("Work Saturdays (Unload)", value=False)
        unload_sunday = st.checkbox("Work Sundays (Unload)", value=False)
        
        # Optimization Parameters
        st.subheader("Optimization Constraints")
        max_loading_days = st.number_input(
            "Max Loading Time (calendar days)",
            min_value=5,
            max_value=30,
            value=14,
            step=1,
            help="Maximum acceptable time to fill a cell"
        )
        
        min_cell_size = st.number_input(
            "Min Cell Size (CY)",
            min_value=50,
            max_value=5000,
            value=900,
            step=50,
            help="Minimum cell size to consider"
        )
        
        max_cell_size = st.number_input(
            "Max Cell Size (CY)",
            min_value=500,
            max_value=10000,
            value=5000,
            step=100,
            help="Maximum cell size to consider"
        )
        
        run_button = st.button("🔍 Optimize Configuration", type="primary", use_container_width=True)
    
    # Main content area
    if run_button:
        with st.spinner("Optimizing cell configuration..."):
            
            # Prepare parameters
            phase_params = {
                'rip_days': rip_days,
                'treat_days': treat_days,
                'dry_days': dry_days
            }
            
            weekend_params = {
                'load_saturday': load_saturday,
                'load_sunday': load_sunday,
                'rip_saturday': rip_saturday,
                'rip_sunday': rip_sunday,
                'treat_saturday': treat_saturday,
                'treat_sunday': treat_sunday,
                'dry_saturday': dry_saturday,
                'dry_sunday': dry_sunday,
                'unload_saturday': unload_saturday,
                'unload_sunday': unload_sunday
            }
            
            # Run optimization
            results_df = optimize_cell_configuration(
                daily_volume,
                daily_load_capacity,
                daily_unload_capacity,
                phase_params,
                weekend_params,
                max_loading_days=max_loading_days,
                min_cell_volume=min_cell_size,
                max_cell_volume=max_cell_size,
                step_size=50
            )
            
            # Store in session state
            st.session_state.results_df = results_df
            st.session_state.selected_config_index = 0 if results_df is not None and len(results_df) > 0 else None
            st.session_state.optimization_run = True
            # Clear any previous schedule when new optimization runs
            st.session_state.schedule_generated = False
            st.session_state.schedule_df = None
            st.session_state.schedule_start = None
            st.session_state.parameters = {
                'daily_volume': daily_volume,
                'daily_load_capacity': daily_load_capacity,
                'daily_unload_capacity': daily_unload_capacity,
                'phase_params': phase_params,
                'weekend_params': weekend_params,
                'rip_days': rip_days,
                'treat_days': treat_days,
                'dry_days': dry_days
            }
    
    # Display results if optimization has been run
    if st.session_state.optimization_run and st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        
        # Retrieve parameters
        daily_volume = st.session_state.parameters['daily_volume']
        daily_load_capacity = st.session_state.parameters['daily_load_capacity']
        daily_unload_capacity = st.session_state.parameters['daily_unload_capacity']
        phase_params = st.session_state.parameters['phase_params']
        weekend_params = st.session_state.parameters['weekend_params']
        rip_days = st.session_state.parameters['rip_days']
        treat_days = st.session_state.parameters['treat_days']
        dry_days = st.session_state.parameters['dry_days']
        
        if results_df is None or len(results_df) == 0:
            st.error("❌ No valid configurations found. Try adjusting constraints.")
            st.info("Suggestions: Increase max cell size, increase loading capacity, or increase max loading days")
        else:
            st.success(f"✅ Found {len(results_df)} viable configurations")
            
            # Summary stats
            st.subheader("📊 Configuration Range Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Cell Sizes",
                    f"{results_df['cell_volume_cy'].min():.0f} - {results_df['cell_volume_cy'].max():.0f} CY",
                    help="Range of cell volumes analyzed"
                )
            
            with col2:
                st.metric(
                    "Number of Cells",
                    f"{results_df['num_cells'].min():.0f} - {results_df['num_cells'].max():.0f}",
                    help="Range of cells needed across configurations"
                )
            
            with col3:
                st.metric(
                    "Equipment Utilization",
                    f"{results_df['equipment_utilization'].min()*100:.0f}% - {results_df['equipment_utilization'].max()*100:.0f}%",
                    help="Range of equipment productivity"
                )
            
            with col4:
                st.metric(
                    "Facility Utilization",
                    f"{results_df['utilization'].min()*100:.0f}% - {results_df['utilization'].max()*100:.0f}%",
                    help="Range of capacity utilization"
                )
            
            # Configuration Matrix
            st.subheader("🔍 Configuration Options Matrix")
            st.markdown("**Compare all viable configurations. Click a row to select it for schedule generation.**")
            
            # Prepare display dataframe with color coding
            display_df = results_df.copy()
            
            # Format columns for display
            display_df['Cell Size (CY)'] = display_df['cell_volume_cy'].astype(int)
            display_df['Cells'] = display_df['num_cells'].astype(int)
            display_df['Total Capacity (CY)'] = display_df['total_capacity_cy'].astype(int)
            display_df['Load Days'] = display_df['load_days'].astype(int)
            display_df['Cycle Days'] = display_df['cycle_days'].astype(int)
            display_df['Days Buffer'] = display_df['days_of_capacity'].round(1)
            display_df['Facility Util %'] = (display_df['utilization'] * 100).round(1)
            display_df['Equipment Util %'] = (display_df['equipment_utilization'] * 100).round(1)
            display_df['Score'] = display_df['score'].round(0).astype(int)
            
            # Create selection column
            display_df.insert(0, 'Select', False)
            
            # Display columns
            display_cols = [
                'Cell Size (CY)', 'Cells', 'Total Capacity (CY)', 
                'Load Days', 'Cycle Days', 'Days Buffer',
                'Facility Util %', 'Equipment Util %', 'Score'
            ]
            
            # Color coding helper
            def color_utilization(val, thresholds=(70, 85)):
                """Color code utilization values"""
                if val >= thresholds[1]:
                    return 'background-color: #90EE90'  # Light green
                elif val >= thresholds[0]:
                    return 'background-color: #FFFFE0'  # Light yellow
                else:
                    return 'background-color: #FFB6C1'  # Light red
            
            # Apply styling
            styled_df = display_df[display_cols].style.applymap(
                color_utilization,
                subset=['Facility Util %', 'Equipment Util %']
            )
            
            # Display the table
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                height=min(400, len(display_df) * 35 + 38)
            )
            
            # Row selection for schedule generation
            st.markdown("---")
            st.subheader("📋 Select Configuration for Schedule")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_index = st.selectbox(
                    "Choose a configuration to generate detailed schedule:",
                    options=range(len(results_df)),
                    format_func=lambda x: f"Config {x+1}: {results_df.iloc[x]['cell_volume_cy']:.0f} CY × {results_df.iloc[x]['num_cells']:.0f} cells (Equip Util: {results_df.iloc[x]['equipment_utilization']*100:.1f}%)",
                    key="config_selector"
                )
                st.session_state.selected_config_index = selected_index
            
            with col2:
                st.metric(
                    "Selected Config",
                    f"#{selected_index + 1}",
                    help="Configuration number selected"
                )
            
            # Display selected configuration details
            selected_config = results_df.iloc[selected_index]
            
            st.markdown("**Selected Configuration Details:**")
            detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
            
            with detail_col1:
                st.metric("Cell Size", f"{selected_config['cell_volume_cy']:.0f} CY")
                st.metric("Number of Cells", f"{selected_config['num_cells']:.0f}")
            
            with detail_col2:
                st.metric("Loading Time", f"{selected_config['load_days']:.0f} days")
                st.metric("Full Cycle", f"{selected_config['cycle_days']:.0f} days")
            
            with detail_col3:
                st.metric("Facility Utilization", f"{selected_config['utilization']*100:.1f}%")
                st.metric("Equipment Utilization", f"{selected_config['equipment_utilization']*100:.1f}%")
            
            with detail_col4:
                st.metric("Total Capacity", f"{selected_config['total_capacity_cy']:.0f} CY")
                st.metric("Days of Buffer", f"{selected_config['days_of_capacity']:.1f} days")
                
            
            # Cycle Time Breakdown for Selected Config
            st.markdown("**Cycle Time Breakdown (Selected Configuration):**")
            cycle_data = {
                'Phase': ['Loading', 'Rip', 'Treatment', 'Drying', 'Unloading', 'TOTAL'],
                'Calendar Days': [
                    selected_config['cycle_breakdown']['load_calendar_days'],
                    selected_config['cycle_breakdown']['rip_calendar_days'],
                    selected_config['cycle_breakdown']['treat_calendar_days'],
                    selected_config['cycle_breakdown']['dry_calendar_days'],
                    selected_config['cycle_breakdown']['unload_calendar_days'],
                    selected_config['cycle_days']
                ]
            }
            st.dataframe(pd.DataFrame(cycle_data), hide_index=True, use_container_width=True)
            
            # Visualizations
            st.subheader("📊 Configuration Comparisons")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Cell Size vs Number", "Utilization Analysis", "Equipment Utilization", "Cycle Time"])
            
            with tab1:
                fig1 = px.scatter(
                    results_df.head(20),
                    x='cell_volume_cy',
                    y='num_cells',
                    size='utilization',
                    color='load_days',
                    hover_data=['cycle_days', 'equipment_utilization'],
                    labels={
                        'cell_volume_cy': 'Cell Size (CY)',
                        'num_cells': 'Number of Cells',
                        'load_days': 'Loading Days',
                        'utilization': 'Facility Utilization',
                        'equipment_utilization': 'Equipment Utilization'
                    },
                    title='Cell Size vs Number of Cells Required',
                    color_continuous_scale='Viridis'
                )
                fig1.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab2:
                fig2 = px.scatter(
                    results_df.head(20),
                    x='cell_volume_cy',
                    y='utilization',
                    size='num_cells',
                    color='num_cells',
                    hover_data=['load_days', 'cycle_days', 'equipment_utilization'],
                    labels={
                        'cell_volume_cy': 'Cell Size (CY)',
                        'utilization': 'Facility Utilization',
                        'num_cells': 'Number of Cells',
                        'equipment_utilization': 'Equipment Utilization'
                    },
                    title='Facility Utilization by Configuration',
                    color_continuous_scale='RdYlGn'
                )
                fig2.add_hline(y=0.85, line_dash="dash", line_color="red", 
                              annotation_text="85% Target")
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab3:
                # Equipment utilization analysis
                fig3 = px.scatter(
                    results_df.head(20),
                    x='cell_volume_cy',
                    y='equipment_utilization',
                    size='num_cells',
                    color='load_days',
                    hover_data=['num_cells', 'cycle_days', 'utilization', 'cell_start_interval'],
                    labels={
                        'cell_volume_cy': 'Cell Size (CY)',
                        'equipment_utilization': 'Equipment Utilization',
                        'num_cells': 'Number of Cells',
                        'load_days': 'Loading Days',
                        'cell_start_interval': 'Cell Start Interval (days)'
                    },
                    title='Equipment Utilization by Configuration (Higher = Less Idle Time)',
                    color_continuous_scale='Blues'
                )
                fig3.add_hline(y=0.75, line_dash="dash", line_color="green", 
                              annotation_text="75% Good Target")
                fig3.add_hline(y=0.85, line_dash="dash", line_color="darkgreen", 
                              annotation_text="85% Excellent")
                st.plotly_chart(fig3, use_container_width=True)
                
                st.info("""
                **Understanding Equipment Utilization:**
                - **85%+**: Excellent - Equipment stays consistently busy with minimal idle time
                - **75-85%**: Good - Solid balance between productivity and flexibility
                - **60-75%**: Fair - Some idle days, consider adding cells for better utilization
                - **<60%**: Poor - Equipment frequently idle, strongly consider more/smaller cells
                
                Higher utilization = loaders/excavators work more days = better ROI on equipment costs
                """)
            
            with tab4:
                # Cycle time breakdown for selected config
                cycle_breakdown = {
                    'Phase': ['Load', 'Rip', 'Treat', 'Dry', 'Unload'],
                    'Days': [
                        selected_config['cycle_breakdown']['load_calendar_days'],
                        selected_config['cycle_breakdown']['rip_calendar_days'],
                        selected_config['cycle_breakdown']['treat_calendar_days'],
                        selected_config['cycle_breakdown']['dry_calendar_days'],
                        selected_config['cycle_breakdown']['unload_calendar_days']
                    ]
                }
                fig4 = px.bar(
                    cycle_breakdown,
                    x='Phase',
                    y='Days',
                    title='Cycle Time Breakdown (Selected Configuration)',
                    labels={'Days': 'Calendar Days'},
                    color='Phase',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig4, use_container_width=True)
            
            # Export options
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            # Prepare export data
            export_df = results_df.copy()
            export_df = export_df.round(2)
            
            # Add summary sheet
            summary_data = {
                'Parameter': [
                    'Daily Incoming Volume (CY)',
                    'Daily Load Capacity (CY)',
                    'Daily Unload Capacity (CY)',
                    'Rip Duration (days)',
                    'Treatment Duration (days)',
                    'Drying Duration (days)',
                    '',
                    'SELECTED CONFIGURATION',
                    'Cell Size (CY)',
                    'Number of Cells',
                    'Total Facility Capacity (CY)',
                    'Cycle Time (days)',
                    'Facility Utilization (%)',
                    'Equipment Utilization (%)'
                ],
                'Value': [
                    daily_volume,
                    daily_load_capacity,
                    daily_unload_capacity,
                    rip_days,
                    treat_days,
                    dry_days,
                    '',
                    '',
                    selected_config['cell_volume_cy'],
                    selected_config['num_cells'],
                    selected_config['total_capacity_cy'],
                    selected_config['cycle_days'],
                    f"{selected_config['utilization']*100:.1f}",
                    f"{selected_config['equipment_utilization']*100:.1f}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            
            # Create Excel file
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                export_df.to_excel(writer, sheet_name='All_Configurations', index=False)
            
            output.seek(0)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            excel_filename = f"facility_optimizer_{timestamp}.xlsx"
            
            with col1:
                st.download_button(
                    label="📥 Download Configuration Analysis",
                    data=output,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"configurations_{timestamp}.csv",
                    mime="text/csv"
                )
            
            # Generate Treatment Schedule
            st.markdown("---")
            st.subheader("📅 Generate Treatment Schedule")
            st.markdown("""
            Generate a detailed day-by-day schedule showing how each treatment cell will operate
            over time based on the selected configuration.
            """)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                schedule_start_date = st.date_input(
                    "Schedule Start Date",
                    value=datetime.now().date(),
                    help="First day of facility operations",
                    key="schedule_start_date_input"
                )
                
                with col2:
                    schedule_days = st.number_input(
                        "Days to Schedule",
                        min_value=30,
                        max_value=365,
                        value=90,
                        step=30,
                        help="Number of days to simulate",
                        key="schedule_days_number"
                    )
                
                if st.button("📊 Generate Detailed Schedule", type="primary", key="generate_schedule_btn"):
                    try:
                        with st.spinner("Generating detailed treatment schedule..."):
                            # Convert date to datetime
                            schedule_start = datetime.combine(schedule_start_date, datetime.min.time())
                            
                            # Generate schedule
                            schedule_df = simulate_facility_schedule(
                                selected_config,
                                daily_volume,
                                daily_load_capacity,
                                daily_unload_capacity,
                                phase_params,
                                weekend_params,
                                schedule_start,
                                int(schedule_days)
                            )
                            
                            # Store in session state
                            st.session_state.schedule_df = schedule_df
                            st.session_state.schedule_generated = True
                            st.session_state.schedule_start = schedule_start
                    except Exception as e:
                        st.error(f"Error generating schedule: {str(e)}")
                        st.error("Please try again or contact support if the issue persists.")
                
                # Display schedule if it has been generated
                if st.session_state.schedule_generated and st.session_state.schedule_df is not None:
                    schedule_df = st.session_state.schedule_df
                    schedule_start = st.session_state.schedule_start
                    selected_config = results_df.iloc[st.session_state.selected_config_index]
                    
                    # Display preview
                    st.success("✅ Schedule Generated!")
                    st.markdown("**Schedule Preview** (first 14 days)")
                    
                    # Create display version
                    display_cols = ['Date', 'DayName']
                    for i in range(1, int(selected_config['num_cells']) + 1):
                        display_cols.append(f'Cell_{i}_Phase')
                    display_cols.extend(['SoilWaiting', 'CumSoilIn', 'CumSoilOut'])
                    
                    preview_df = schedule_df[display_cols].head(14).copy()
                    preview_df['Date'] = preview_df['Date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
                    
                    # Create formatted Excel export
                    st.markdown("**Export Formatted Schedule**")
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_schedule = BytesIO()
                    with pd.ExcelWriter(output_schedule, engine='openpyxl') as writer:
                            # Write schedule
                            schedule_export = schedule_df.copy()
                            schedule_export.to_excel(writer, sheet_name='Schedule', index=False)
                            
                            # Write summary
                            schedule_summary = {
                                'Metric': [
                                    'Start Date',
                                    'Days Simulated',
                                    'Cell Size (CY)',
                                    'Number of Cells',
                                    'Daily Incoming Volume (CY)',
                                    'Total Soil Loaded (CY)',
                                    'Total Soil Unloaded (CY)',
                                    'Soil Waiting (End)',
                                ],
                                'Value': [
                                    schedule_start.strftime('%Y-%m-%d'),
                                    len(schedule_df),
                                    selected_config['cell_volume_cy'],
                                    int(selected_config['num_cells']),
                                    daily_volume,
                                    schedule_df['CumSoilIn'].iloc[-1],
                                    schedule_df['CumSoilOut'].iloc[-1],
                                    schedule_df['SoilWaiting'].iloc[-1]
                                ]
                            }
                            pd.DataFrame(schedule_summary).to_excel(writer, sheet_name='Summary', index=False)
                            
                            # Apply formatting to Schedule sheet
                            from openpyxl.styles import PatternFill, Border, Side, Alignment, Font
                            from openpyxl.utils import get_column_letter
                            
                            workbook = writer.book
                            worksheet = writer.sheets['Schedule']
                            
                            # Define colors matching original
                            phase_colors = {
                                'Loading': '#8ED973',
                                'Rip': '#83CCEB',
                                'Treat': '#FFC000',
                                'Dry': '#F2CEEF',
                                'ReadyToUnload': '#00B0F0',
                                'Empty': '#FFFFFF'
                            }
                            
                            sunday_fill = PatternFill(start_color='FFFFFF00', end_color='FFFFFF00', fill_type='solid')
                            
                            thin_border = Border(
                                left=Side(style='thin', color='000000'),
                                right=Side(style='thin', color='000000'),
                                top=Side(style='thin', color='000000'),
                                bottom=Side(style='thin', color='000000')
                            )
                            aptos_font = Font(name='Aptos Narrow', size=10)
                            aptos_font_bold = Font(name='Aptos Narrow', size=10, bold=True)
                            center_aligned = Alignment(horizontal='center', vertical='center')
                            
                            # Find columns
                            phase_columns = []
                            date_column_idx = None
                            dayname_column_idx = None
                            
                            for col_idx, col_name in enumerate(schedule_df.columns, start=1):
                                if 'Phase' in col_name:
                                    phase_columns.append((col_idx, col_name))
                                if col_name == 'Date':
                                    date_column_idx = col_idx
                                if col_name == 'DayName':
                                    dayname_column_idx = col_idx
                            
                            # Apply formatting
                            for row_idx in range(1, len(schedule_df) + 2):
                                is_sunday = False
                                
                                if row_idx > 1:
                                    if dayname_column_idx:
                                        day_name_cell = worksheet.cell(row=row_idx, column=dayname_column_idx)
                                        is_sunday = str(day_name_cell.value) == 'Sunday'
                                
                                for col_idx in range(1, len(schedule_df.columns) + 1):
                                    cell = worksheet.cell(row=row_idx, column=col_idx)
                                    is_phase_column = any(col_idx == phase_col_idx for phase_col_idx, _ in phase_columns)
                                    
                                    cell.border = thin_border
                                    cell.font = aptos_font_bold if row_idx == 1 else aptos_font
                                    
                                    if col_idx == date_column_idx and row_idx > 1:
                                        cell.number_format = 'M/D/YYYY'
                                    
                                    if is_sunday and not is_phase_column and row_idx > 1 and col_idx != date_column_idx:
                                        cell.fill = sunday_fill
                                    
                                    if row_idx > 1:
                                        for phase_col_idx, phase_col_name in phase_columns:
                                            if col_idx == phase_col_idx:
                                                cell.alignment = center_aligned
                                                phase_value = str(cell.value) if cell.value else ''
                                                
                                                for phase_name, color in phase_colors.items():
                                                    if phase_name in phase_value:
                                                        hex_color = color.lstrip('#')
                                                        openpyxl_color = 'FF' + hex_color
                                                        cell.fill = PatternFill(start_color=openpyxl_color,
                                                                              end_color=openpyxl_color,
                                                                              fill_type='solid')
                                                        break
                            
                            # Auto-adjust column widths
                            for column_cells in worksheet.columns:
                                length = max(len(str(cell.value) if cell.value else "") for cell in column_cells)
                                worksheet.column_dimensions[get_column_letter(column_cells[0].column)].width = min(length + 2, 50)
                    
                    output_schedule.seek(0)
                    
                    schedule_filename = f"treatment_schedule_{timestamp}.xlsx"
                    st.download_button(
                        label="📥 Download Formatted Schedule",
                        data=output_schedule,
                        file_name=schedule_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_schedule"
                    )
    
    else:
        # Initial state - show instructions
        st.info("👈 Configure your facility parameters in the sidebar and click **Optimize Configuration**")
        
        st.markdown("""
        ### How This Tool Works
        
        This optimizer helps you design a continuous-flow soil remediation facility by:
        
        1. **Analyzing Your Requirements**: Input your daily incoming soil volume and operational parameters
        2. **Calculating Cycle Times**: Determines how long each treatment cell takes to complete a full cycle
        3. **Finding All Viable Configurations**: Generates a matrix of options balancing:
           - Number of cells (capital cost)
           - Cell size (volume in CY)
           - Facility utilization (capacity management)
           - **Equipment utilization (keeping loaders/excavators busy daily)**
           - Loading time (operational constraints)
        4. **You Choose**: Select the configuration that best fits your needs
        5. **Generating Schedules**: Creates detailed day-by-day treatment schedules for your selected configuration
        
        ### Key Features
        
        - **Configuration Matrix**: Shows ALL viable options, not just one "optimal" recommendation
        - **Comparison Tools**: Sort and filter by any metric - cell size, number, utilization, etc.
        - **Equipment Productivity**: Tracks how often loaders and excavators stay busy
        - **Color-Coded Utilization**: Green = good, yellow = fair, red = poor
        - **Weekend Scheduling**: Customizable work schedules by phase
        - **Detailed Schedules**: Day-by-day tracking of all cells with color-coded phases
        - **Excel Export**: Formatted reports with all configurations and selected config details
        
        ### Why Equipment Utilization Matters
        
        **Idle equipment = wasted money.** The analysis shows equipment utilization because:
        - Loaders and excavators are expensive assets (purchase or rental)
        - Operators are paid whether equipment is working or idle
        - Smaller, more numerous cells = more consistent daily work
        - Larger cells = longer loading periods = more idle days between operations
        
        **Target**: 75-85%+ equipment utilization means your team works productively most days.
        
        ### Key Tradeoffs
        
        - **More cells** = Higher capital cost, BUT better equipment utilization and surge capacity
        - **Larger cells** = Fewer cells needed, BUT longer loading times and more equipment idle days
        - **Facility utilization** = 80-90% is good for capacity management
        - **Equipment utilization** = 75-85%+ minimizes idle time and maximizes productivity
        
        ### Workflow
        
        1. Set your daily incoming volume
        2. Configure equipment capacities
        3. Set treatment phase durations
        4. Define weekend working schedules
        5. Run the analysis
        6. **Review the configuration matrix** - all options shown
        7. **Select your preferred configuration** - you decide what's best
        8. Generate detailed treatment schedule for selected config
        9. Export results and schedules
        """)


if __name__ == "__main__":
    main()
