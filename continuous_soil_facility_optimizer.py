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


def calculate_cell_dimensions(cell_volume_cy, depth_ft):
    """Calculate cell length and width from volume and depth
    Assumes square cells for simplicity (L = W)
    """
    # Convert CY to cubic feet
    volume_cf = cell_volume_cy * 27
    
    # For a square cell: Volume = L * W * D = L^2 * D
    # Therefore: L = sqrt(Volume / D)
    side_length_ft = math.sqrt(volume_cf / depth_ft)
    
    return {
        'Length_ft': side_length_ft,
        'Width_ft': side_length_ft,
        'Depth_ft': depth_ft,
        'Volume_CY': cell_volume_cy,
        'Volume_CF': volume_cf,
        'Area_SF': side_length_ft ** 2
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


def optimize_cell_configuration(daily_volume_cy, depth_ft, daily_load_capacity,
                                daily_unload_capacity, phase_params, weekend_params,
                                max_loading_days=14, min_cell_volume=100, 
                                max_cell_volume=5000, step_size=50):
    """Find optimal cell configuration across a range of cell sizes"""
    
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
        
        # Calculate dimensions
        dimensions = calculate_cell_dimensions(cell_volume, depth_ft)
        
        # Calculate capacity metrics
        total_facility_capacity = cell_volume * cells_info['min_cells_with_buffer']
        days_of_capacity = total_facility_capacity / daily_volume_cy
        
        # Calculate utilization
        # How much soil can we process per day on average?
        daily_throughput = cell_volume / cycle_info['total_calendar_days']
        utilization = daily_volume_cy / (daily_throughput * cells_info['min_cells_with_buffer'])
        
        # Score: prefer fewer cells, but penalize poor utilization
        # Lower score is better
        score = cells_info['min_cells_with_buffer'] * 100 + (1 - utilization) * 1000
        
        results.append({
            'cell_volume_cy': cell_volume,
            'num_cells': cells_info['min_cells_with_buffer'],
            'length_ft': dimensions['Length_ft'],
            'width_ft': dimensions['Width_ft'],
            'depth_ft': dimensions['Depth_ft'],
            'area_sf': dimensions['Area_SF'],
            'load_days': cycle_info['load_calendar_days'],
            'cycle_days': cycle_info['total_calendar_days'],
            'total_capacity_cy': total_facility_capacity,
            'days_of_capacity': days_of_capacity,
            'utilization': utilization,
            'daily_throughput': daily_throughput * cells_info['min_cells_with_buffer'],
            'score': score,
            'cycle_breakdown': cycle_info
        })
    
    if not results:
        return None
    
    # Sort by score (lower is better)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score').reset_index(drop=True)
    
    return results_df


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
        
        # Cell Depth
        st.subheader("Cell Design")
        cell_depth = st.number_input(
            "Treatment Cell Depth (feet)",
            min_value=1.0,
            max_value=20.0,
            value=4.0,
            step=0.5,
            help="Desired depth for treatment cells"
        )
        
        # Equipment Capacity
        st.subheader("Equipment Capacity")
        daily_load_capacity = st.number_input(
            "Daily Loading Capacity (CY/day)",
            min_value=50,
            max_value=2000,
            value=300,
            step=25,
            help="Maximum soil that can be loaded per day"
        )
        
        daily_unload_capacity = st.number_input(
            "Daily Unloading Capacity (CY/day)",
            min_value=50,
            max_value=2000,
            value=300,
            step=25,
            help="Maximum soil that can be unloaded per day"
        )
        
        # Phase Durations
        st.subheader("Treatment Phase Durations (Work Days)")
        rip_days = st.number_input("Rip Duration (days)", min_value=1, value=3, step=1)
        treat_days = st.number_input("Treatment Duration (days)", min_value=1, value=14, step=1)
        dry_days = st.number_input("Drying Duration (days)", min_value=1, value=7, step=1)
        
        # Weekend Working
        st.subheader("Weekend Operations")
        
        st.markdown("**Loading**")
        load_saturday = st.checkbox("Work Saturdays (Load)", value=True)
        load_sunday = st.checkbox("Work Sundays (Load)", value=False)
        
        st.markdown("**Rip**")
        rip_saturday = st.checkbox("Work Saturdays (Rip)", value=True)
        rip_sunday = st.checkbox("Work Sundays (Rip)", value=False)
        
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
        unload_saturday = st.checkbox("Work Saturdays (Unload)", value=True)
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
            max_value=1000,
            value=100,
            step=50,
            help="Minimum cell size to consider"
        )
        
        max_cell_size = st.number_input(
            "Max Cell Size (CY)",
            min_value=500,
            max_value=10000,
            value=3000,
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
                cell_depth,
                daily_load_capacity,
                daily_unload_capacity,
                phase_params,
                weekend_params,
                max_loading_days=max_loading_days,
                min_cell_volume=min_cell_size,
                max_cell_volume=max_cell_size,
                step_size=50
            )
            
            if results_df is None or len(results_df) == 0:
                st.error("❌ No valid configurations found. Try adjusting constraints.")
                st.info("Suggestions: Increase max cell size, increase loading capacity, or increase max loading days")
            else:
                # Display optimal configuration
                st.success("✅ Optimization Complete")
                
                optimal = results_df.iloc[0]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Optimal Cell Size",
                        f"{optimal['cell_volume_cy']:.0f} CY",
                        help="Recommended cell capacity"
                    )
                
                with col2:
                    st.metric(
                        "Number of Cells",
                        f"{optimal['num_cells']:.0f}",
                        help="Minimum cells needed for continuous operation"
                    )
                
                with col3:
                    st.metric(
                        "Cell Dimensions",
                        f"{optimal['length_ft']:.1f}' × {optimal['width_ft']:.1f}'",
                        help=f"Length × Width (at {optimal['depth_ft']:.1f}' depth)"
                    )
                
                with col4:
                    st.metric(
                        "Utilization",
                        f"{optimal['utilization']*100:.1f}%",
                        help="Facility utilization rate"
                    )
                
                # Detailed breakdown
                st.subheader("📊 Optimal Configuration Details")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Cell Specifications**")
                    specs_data = {
                        'Parameter': [
                            'Cell Volume',
                            'Cell Length',
                            'Cell Width',
                            'Cell Depth',
                            'Cell Area',
                            'Number of Cells',
                            'Total Facility Capacity'
                        ],
                        'Value': [
                            f"{optimal['cell_volume_cy']:.0f} CY",
                            f"{optimal['length_ft']:.1f} ft",
                            f"{optimal['width_ft']:.1f} ft",
                            f"{optimal['depth_ft']:.1f} ft",
                            f"{optimal['area_sf']:.0f} sq ft",
                            f"{optimal['num_cells']:.0f}",
                            f"{optimal['total_capacity_cy']:.0f} CY"
                        ]
                    }
                    st.dataframe(pd.DataFrame(specs_data), hide_index=True, use_container_width=True)
                
                with col2:
                    st.markdown("**Cycle Time Breakdown**")
                    cycle_data = {
                        'Phase': [
                            'Loading',
                            'Rip',
                            'Treatment',
                            'Drying',
                            'Unloading',
                            'TOTAL CYCLE'
                        ],
                        'Calendar Days': [
                            optimal['cycle_breakdown']['load_calendar_days'],
                            optimal['cycle_breakdown']['rip_calendar_days'],
                            optimal['cycle_breakdown']['treat_calendar_days'],
                            optimal['cycle_breakdown']['dry_calendar_days'],
                            optimal['cycle_breakdown']['unload_calendar_days'],
                            optimal['cycle_days']
                        ]
                    }
                    st.dataframe(pd.DataFrame(cycle_data), hide_index=True, use_container_width=True)
                
                # Performance metrics
                st.subheader("📈 Performance Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Days of Capacity",
                        f"{optimal['days_of_capacity']:.1f} days",
                        help="How many days of incoming volume the facility can hold"
                    )
                
                with col2:
                    st.metric(
                        "Daily Throughput",
                        f"{optimal['daily_throughput']:.0f} CY/day",
                        help="Maximum average daily processing capacity"
                    )
                
                with col3:
                    surplus = optimal['daily_throughput'] - daily_volume
                    st.metric(
                        "Capacity Surplus",
                        f"{surplus:.0f} CY/day",
                        help="Extra capacity beyond daily incoming volume",
                        delta=f"{(surplus/daily_volume)*100:.1f}%"
                    )
                
                # Alternative configurations
                st.subheader("🔄 Alternative Configurations")
                st.markdown("Top 10 alternative configurations ranked by optimization score")
                
                display_df = results_df.head(10).copy()
                display_df['Cell Size (CY)'] = display_df['cell_volume_cy'].astype(int)
                display_df['Cells'] = display_df['num_cells'].astype(int)
                display_df['Dimensions (L×W)'] = display_df.apply(
                    lambda x: f"{x['length_ft']:.1f}' × {x['width_ft']:.1f}'", axis=1
                )
                display_df['Load Days'] = display_df['load_days'].astype(int)
                display_df['Cycle Days'] = display_df['cycle_days'].astype(int)
                display_df['Utilization'] = (display_df['utilization'] * 100).round(1).astype(str) + '%'
                display_df['Total Capacity (CY)'] = display_df['total_capacity_cy'].astype(int)
                
                st.dataframe(
                    display_df[[
                        'Cell Size (CY)', 'Cells', 'Dimensions (L×W)', 
                        'Load Days', 'Cycle Days', 'Utilization', 'Total Capacity (CY)'
                    ]],
                    hide_index=True,
                    use_container_width=True
                )
                
                # Visualizations
                st.subheader("📊 Configuration Analysis")
                
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["Cell Size vs Number", "Utilization Analysis", "Cycle Time"])
                
                with tab1:
                    fig1 = px.scatter(
                        results_df.head(20),
                        x='cell_volume_cy',
                        y='num_cells',
                        size='utilization',
                        color='load_days',
                        hover_data=['cycle_days', 'utilization'],
                        labels={
                            'cell_volume_cy': 'Cell Size (CY)',
                            'num_cells': 'Number of Cells',
                            'load_days': 'Loading Days',
                            'utilization': 'Utilization'
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
                        hover_data=['load_days', 'cycle_days'],
                        labels={
                            'cell_volume_cy': 'Cell Size (CY)',
                            'utilization': 'Utilization',
                            'num_cells': 'Number of Cells'
                        },
                        title='Facility Utilization by Configuration',
                        color_continuous_scale='RdYlGn'
                    )
                    fig2.add_hline(y=0.85, line_dash="dash", line_color="red", 
                                  annotation_text="85% Target")
                    st.plotly_chart(fig2, use_container_width=True)
                
                with tab3:
                    # Cycle time breakdown for optimal config
                    cycle_breakdown = {
                        'Phase': ['Load', 'Rip', 'Treat', 'Dry', 'Unload'],
                        'Days': [
                            optimal['cycle_breakdown']['load_calendar_days'],
                            optimal['cycle_breakdown']['rip_calendar_days'],
                            optimal['cycle_breakdown']['treat_calendar_days'],
                            optimal['cycle_breakdown']['dry_calendar_days'],
                            optimal['cycle_breakdown']['unload_calendar_days']
                        ]
                    }
                    fig3 = px.bar(
                        cycle_breakdown,
                        x='Phase',
                        y='Days',
                        title='Cycle Time Breakdown (Optimal Configuration)',
                        labels={'Days': 'Calendar Days'},
                        color='Phase',
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Export options
                st.subheader("💾 Export Results")
                
                # Prepare export data
                export_df = results_df.copy()
                export_df = export_df.round(2)
                
                # Add summary sheet
                summary_data = {
                    'Parameter': [
                        'Daily Incoming Volume (CY)',
                        'Cell Depth (ft)',
                        'Daily Load Capacity (CY)',
                        'Daily Unload Capacity (CY)',
                        'Rip Duration (days)',
                        'Treatment Duration (days)',
                        'Drying Duration (days)',
                        '',
                        'OPTIMAL CONFIGURATION',
                        'Recommended Cell Size (CY)',
                        'Number of Cells',
                        'Cell Length (ft)',
                        'Cell Width (ft)',
                        'Total Facility Capacity (CY)',
                        'Cycle Time (days)',
                        'Utilization (%)'
                    ],
                    'Value': [
                        daily_volume,
                        cell_depth,
                        daily_load_capacity,
                        daily_unload_capacity,
                        rip_days,
                        treat_days,
                        dry_days,
                        '',
                        '',
                        optimal['cell_volume_cy'],
                        optimal['num_cells'],
                        f"{optimal['length_ft']:.1f}",
                        f"{optimal['width_ft']:.1f}",
                        optimal['total_capacity_cy'],
                        optimal['cycle_days'],
                        f"{optimal['utilization']*100:.1f}"
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
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Download Excel Report",
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
    
    else:
        # Initial state - show instructions
        st.info("👈 Configure your facility parameters in the sidebar and click **Optimize Configuration**")
        
        st.markdown("""
        ### How This Tool Works
        
        This optimizer helps you design a continuous-flow soil remediation facility by:
        
        1. **Analyzing Your Requirements**: Input your daily incoming soil volume and operational parameters
        2. **Calculating Cycle Times**: Determines how long each treatment cell takes to complete a full cycle
        3. **Optimizing Configuration**: Finds the best balance between:
           - Number of cells (capital cost)
           - Cell size (operational efficiency)
           - Utilization (capacity management)
           - Loading time (operational constraints)
        
        ### Key Considerations
        
        - **More cells** = Higher capital cost, but better surge capacity
        - **Larger cells** = Fewer cells needed, but longer loading times
        - **Optimal utilization** = 80-90% for good balance of cost and flexibility
        
        ### Workflow
        
        1. Set your daily incoming volume
        2. Define your cell depth preference
        3. Configure equipment capacities
        4. Set treatment phase durations
        5. Define weekend working schedules
        6. Run the optimization
        7. Review recommendations and alternatives
        """)


if __name__ == "__main__":
    main()
