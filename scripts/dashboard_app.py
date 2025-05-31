import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, time
import logging
from data_processing import (
    fetch_and_preprocess, 
    get_dataframe_from_result, 
    get_actual_game_count, 
    get_actual_win_stats,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Lichess Chess Dashboard",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def display_data_summary(csv_dict):
    """Display summary statistics of loaded data with improved accuracy"""
    if not csv_dict:
        st.warning("No data available for summary")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Get total games count
        total_games = get_actual_game_count(csv_dict)
        st.metric("Total Games", total_games if total_games > 0 else "N/A")
    
    with col2:
        # Get win statistics
        win_stats = get_actual_win_stats(csv_dict)
        if win_stats and win_stats.get("win_rate", 0) > 0:
            st.metric("Win Rate", f"{win_stats['win_rate']:.1f}%")
        else:
            st.metric("Win Rate", "N/A")
    
    with col3:
        # Average rating from Elo data
        elo_data = get_dataframe_from_result(csv_dict, "elo_progression")
        if elo_data is not None and not elo_data.empty and 'UserElo' in elo_data.columns:
            try:
                avg_rating = elo_data['UserElo'].mean()
                if pd.notna(avg_rating):
                    st.metric("Avg Rating", f"{avg_rating:.0f}")
                else:
                    st.metric("Avg Rating", "N/A")
            except Exception as e:
                st.metric("Avg Rating", "N/A")
        else:
            st.metric("Avg Rating", "N/A")
    
    with col4:
        # Date range from Elo data
        if elo_data is not None and not elo_data.empty and 'Date' in elo_data.columns:
            try:
                elo_data_copy = elo_data.copy()
                elo_data_copy['Date'] = pd.to_datetime(elo_data_copy['Date'])
                if len(elo_data_copy) > 1:
                    date_range = (elo_data_copy['Date'].max() - elo_data_copy['Date'].min()).days
                    st.metric("Date Range", f"{date_range} days")
                else:
                    st.metric("Date Range", "1 day")
            except Exception as e:
                st.metric("Date Range", "N/A")
        else:
            st.metric("Date Range", "N/A")

def create_elo_progression_chart(csv_dict):
    """Create Elo progression chart with polynomial approximation"""
    # Use helper function instead of direct CSV reading
    elo_df = get_dataframe_from_result(csv_dict, 'elo_progression')
    
    if elo_df is None or len(elo_df) == 0:
        return None
    
    try:
        # Ensure date column is datetime
        if 'Date' in elo_df.columns:
            elo_df['Date'] = pd.to_datetime(elo_df['Date'])
            elo_df = elo_df.sort_values(by='Date')
        
        fig = go.Figure()
        
        # Add main Elo line
        fig.add_trace(go.Scatter(
            x=elo_df['Date'],
            y=elo_df['UserElo'],
            mode='lines+markers',
            name='Rating',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4),
            hovertemplate='Date: %{x}<br>Rating: %{y}<extra></extra>'
        ))
        
        # Add polynomial approximation if available
        poly_df = get_dataframe_from_result(csv_dict, 'poly_approx')
        if poly_df is not None and len(poly_df) > 0:
            poly_df['date'] = pd.to_datetime(poly_df['date'])
            fig.add_trace(go.Scatter(
                x=poly_df['date'],
                y=poly_df['rating'],
                mode='lines',
                name='Polynomial Trend',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='Trend: %{y}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Elo Progression Over Time',
            xaxis_title='Date',
            yaxis_title='Rating',
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating Elo chart: {e}")
        return None

def create_win_rate_chart(csv_dict):
    """Create win rate analysis chart"""
    # Use helper function instead of direct CSV reading
    win_rate_df = get_dataframe_from_result(csv_dict, 'win_rates')
    
    if win_rate_df is None or len(win_rate_df) == 0:
        st.warning("No win rate data available")
        return None
    
    try:
        # Prepare data for the stacked bar chart
        if all(col in win_rate_df.columns for col in ['Color', 'Win Rate (%)', 'Draw Rate (%)', 'Loss Rate (%)']):
            win_rate_stacked = win_rate_df.melt(
                id_vars=["Color"],
                value_vars=["Win Rate (%)", "Draw Rate (%)", "Loss Rate (%)"],
                var_name="Result Type",
                value_name="Percentage"
            )
            
            fig = px.bar(
                win_rate_stacked,
                x="Color",
                y="Percentage",
                color="Result Type",
                title="Win, Draw, and Loss Rates by Color",
                text="Percentage",
                labels={"Percentage": "Percentage (%)", "Color": "Player Color"},
                color_discrete_map={
                    "Win Rate (%)": "#2E8B57",
                    "Draw Rate (%)": "#FFD700", 
                    "Loss Rate (%)": "#DC143C"
                }
            )
            
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
            fig.update_layout(height=400, showlegend=True)
            return fig
        else:
            st.warning(f"Missing required columns. Available: {list(win_rate_df.columns)}")
            return None
    
    except Exception as e:
        st.error(f"Error creating win rate chart: {e}")
        return None

def create_opening_charts(csv_dict):
    """Create opening performance charts"""
    # Use helper function instead of direct CSV reading
    opening_df = get_dataframe_from_result(csv_dict, 'opening_performance')
    
    if opening_df is None or len(opening_df) == 0:
        st.warning("No opening performance data available")
        return None, None, None, None
    
    try:
        charts = []
        
        # White openings by games count
        if 'UserColor' in opening_df.columns and 'TotalGames' in opening_df.columns:
            white_openings = opening_df[opening_df["UserColor"] == "white"].nlargest(10, "TotalGames")
            if len(white_openings) > 0:
                fig_white_count = px.bar(
                    white_openings,
                    x="OpeningName",
                    y="TotalGames",
                    title="Top 10 Openings by Games Played (White)",
                    labels={"TotalGames": "Number of Games", "OpeningName": "Opening"},
                    text="TotalGames",
                    color="TotalGames",
                    color_continuous_scale="Blues"
                )
                fig_white_count.update_xaxes(tickangle=45)
                fig_white_count.update_traces(textposition='outside')
                fig_white_count.update_layout(height=500, showlegend=False)
                charts.append(fig_white_count)
            else:
                charts.append(None)
        else:
            st.warning(f"Missing columns for white openings. Need: UserColor, TotalGames. Have: {list(opening_df.columns)}")
            charts.append(None)
        
        # Black openings by games count
        if 'UserColor' in opening_df.columns and 'TotalGames' in opening_df.columns:
            black_openings = opening_df[opening_df["UserColor"] == "black"].nlargest(10, "TotalGames")
            if len(black_openings) > 0:
                fig_black_count = px.bar(
                    black_openings,
                    x="OpeningName",
                    y="TotalGames",
                    title="Top 10 Openings by Games Played (Black)",
                    labels={"TotalGames": "Number of Games", "OpeningName": "Opening"},
                    text="TotalGames",
                    color="TotalGames",
                    color_continuous_scale="Reds"
                )
                fig_black_count.update_xaxes(tickangle=45)
                fig_black_count.update_traces(textposition='outside')
                fig_black_count.update_layout(height=500, showlegend=False)
                charts.append(fig_black_count)
            else:
                charts.append(None)
        else:
            st.warning(f"Missing columns for black openings. Need: UserColor, TotalGames. Have: {list(opening_df.columns)}")
            charts.append(None)
        
        # White opening performance (Win/Draw/Loss)
        if all(col in opening_df.columns for col in ['UserColor', 'TotalGames', 'Win', 'Draw', 'Loss']):
            white_perf = opening_df[opening_df["UserColor"] == "white"].nlargest(10, "TotalGames")
            if len(white_perf) > 0:
                # Reshape data for stacked bar chart
                white_perf_melted = white_perf.melt(
                    id_vars=['OpeningName', 'TotalGames'],
                    value_vars=['Win', 'Draw', 'Loss'],
                    var_name='Result',
                    value_name='Percentage'
                )
                
                fig_white_perf = px.bar(
                    white_perf_melted,
                    x="OpeningName",
                    y="Percentage",
                    color="Result",
                    title="Top 10 Openings Performance as White (%)",
                    labels={"Percentage": "Percentage (%)", "OpeningName": "Opening"},
                    text="Percentage",
                    color_discrete_map={
                        "Win": "#2E8B57",
                        "Draw": "#FFD700", 
                        "Loss": "#DC143C"
                    }
                )
                fig_white_perf.update_xaxes(tickangle=45)
                fig_white_perf.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
                fig_white_perf.update_layout(height=500, showlegend=True)
                charts.append(fig_white_perf)
            else:
                charts.append(None)
        else:
            st.warning(f"Missing columns for white performance. Need: Win, Draw, Loss. Have: {list(opening_df.columns)}")
            charts.append(None)
        
        # Black opening performance (Win/Draw/Loss)
        if all(col in opening_df.columns for col in ['UserColor', 'TotalGames', 'Win', 'Draw', 'Loss']):
            black_perf = opening_df[opening_df["UserColor"] == "black"].nlargest(10, "TotalGames")
            if len(black_perf) > 0:
                # Reshape data for stacked bar chart
                black_perf_melted = black_perf.melt(
                    id_vars=['OpeningName', 'TotalGames'],
                    value_vars=['Win', 'Draw', 'Loss'],
                    var_name='Result',
                    value_name='Percentage'
                )
                
                fig_black_perf = px.bar(
                    black_perf_melted,
                    x="OpeningName",
                    y="Percentage",
                    color="Result",
                    title="Top 10 Openings Performance as Black (%)",
                    labels={"Percentage": "Percentage (%)", "OpeningName": "Opening"},
                    text="Percentage",
                    color_discrete_map={
                        "Win": "#2E8B57",
                        "Draw": "#FFD700", 
                        "Loss": "#DC143C"
                    }
                )
                fig_black_perf.update_xaxes(tickangle=45)
                fig_black_perf.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
                fig_black_perf.update_layout(height=500, showlegend=True)
                charts.append(fig_black_perf)
            else:
                charts.append(None)
        else:
            st.warning(f"Missing columns for black performance. Need: Win, Draw, Loss. Have: {list(opening_df.columns)}")
            charts.append(None)
        
        return charts[0], charts[1], charts[2], charts[3]
    
    except Exception as e:
        st.error(f"Error creating opening charts: {e}")
        return None, None, None, None

def create_game_results_pie_chart(csv_dict):
    """Create a pie chart for game results"""
    game_results_df = get_dataframe_from_result(csv_dict, 'game_results')
    
    if game_results_df is None or len(game_results_df) == 0:
        return None
    
    try:
        if 'Result' in game_results_df.columns and 'Count' in game_results_df.columns:
            fig = px.pie(
                game_results_df,
                values='Count',
                names='Result',
                title='Game Results Distribution',
                color_discrete_map={
                    'Win': '#2E8B57',
                    'Draw': '#FFD700',
                    'Loss': '#DC143C'
                }
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            return fig
    except Exception as e:
        st.error(f"Error creating game results chart: {e}")
    
    return None

def main():
    st.title("♟️ Lichess Chess Dashboard")
    st.markdown("---")
    
    # Sidebar for user input
    st.sidebar.header("📝 Dashboard Settings")
    
    # User input
    username = st.sidebar.text_input("Lichess Username", value="", placeholder="Enter your username")
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Game variant selection
    variant = st.sidebar.selectbox(
        "Game Variant",
        ["blitz", "bullet", "rapid", "classical", "correspondence"],
        index=0
    )
    
    # Additional filters
    st.sidebar.subheader("🔍 Filters")
    rated_filter = st.sidebar.selectbox("Rated Games", ["All", "Rated Only", "Casual Only"], index=0)
    analyzed_filter = st.sidebar.selectbox("Computer Analysis", ["All", "Analyzed Only", "Not Analyzed"], index=0)
    
    # Convert filters to boolean values
    rated = None if rated_filter == "All" else (True if rated_filter == "Rated Only" else False)
    analyzed = None if analyzed_filter == "All" else (True if analyzed_filter == "Analyzed Only" else False)
    
    # Process button
    if st.sidebar.button("🚀 Load Data", type="primary"):
        if not username:
            st.error("Please enter a Lichess username")
            return
        
        # Convert dates to milliseconds - FIXED VERSION
        try:
            # Convert date objects to datetime objects
            start_datetime = datetime.combine(start_date, time.min)  # Start of day (00:00:00)
            end_datetime = datetime.combine(end_date, time.max)      # End of day (23:59:59)
            
            # Convert to milliseconds since epoch
            since = int(start_datetime.timestamp() * 1000)
            until = int(end_datetime.timestamp() * 1000)
            
        except Exception as e:
            st.error(f"❌ Error converting dates: {str(e)}")
            return
        
        # Progress bar replacing spinner
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔗 Connecting to Lichess API...")
            progress_bar.progress(20)
            
            csv_dict = fetch_and_preprocess(
                username=username,
                since=since,
                until=until,
                variant=variant,
                analyzed=analyzed,
                rated=rated
            )
            
            status_text.text("📊 Processing game data...")
            progress_bar.progress(80)
            
            if csv_dict:
                progress_bar.progress(100)
                status_text.text("✅ Data loading complete!")
                
                st.session_state['csv_dict'] = csv_dict
                st.session_state['username'] = username
                st.success(f"✅ Data loaded successfully for {username}!")
                
                # Show some basic stats immediately
                total_games = get_actual_game_count(csv_dict)
                if total_games > 0:
                    st.info(f"📊 Found {total_games} games in the selected date range")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
            else:
                progress_bar.empty()
                status_text.empty()
                st.error("❌ No data found for the specified criteria")
                st.info("💡 Try adjusting the date range or filters")
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error loading data: {str(e)}")
            logger.error(f"Error in main: {e}")
            
            # Additional troubleshooting info
            with st.expander("🔧 Troubleshooting", expanded=False):
                st.text(f"Username: {username}")
                st.text(f"Date range: {start_date} to {end_date}")
                st.text(f"Variant: {variant}")
                st.text(f"Filters: Rated={rated}, Analyzed={analyzed}")
                st.text(f"Error details: {str(e)}")
    
    # Display data if available
    if 'csv_dict' in st.session_state and st.session_state['csv_dict']:
        csv_dict = st.session_state['csv_dict']
        username = st.session_state.get('username', 'Unknown')
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Data Summary", 
            "📊 Elo Progression", 
            "🎯 Win Rates", 
            "♟️ Opening Analysis",
            "📋 Raw Data"
        ])
        
        # Tab 1: Data Summary
        with tab1:
            st.subheader(f"📈 Data Summary for {username}")
            display_data_summary(csv_dict)
            
            # Game results pie chart
            st.subheader("🎯 Game Results Distribution")
            pie_chart = create_game_results_pie_chart(csv_dict)
            if pie_chart:
                st.plotly_chart(pie_chart, use_container_width=True)
            else:
                st.warning("No game results data available for pie chart")
        
        # Tab 2: Elo Progression
        with tab2:
            st.subheader("📊 Elo Progression")
            elo_chart = create_elo_progression_chart(csv_dict)
            if elo_chart:
                st.plotly_chart(elo_chart, use_container_width=True)
                
                # Additional Elo statistics
                elo_data = get_dataframe_from_result(csv_dict, "elo_progression")
                if elo_data is not None and not elo_data.empty:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Highest Rating", f"{elo_data['UserElo'].max():.0f}")
                    with col2:
                        st.metric("Lowest Rating", f"{elo_data['UserElo'].min():.0f}")
                    with col3:
                        rating_change = elo_data['UserElo'].iloc[-1] - elo_data['UserElo'].iloc[0]
                        st.metric("Net Change", f"{rating_change:+.0f}")
            else:
                st.warning("No Elo progression data available")
        
        # Tab 3: Win Rates
        with tab3:
            st.subheader("🎯 Win Rate Analysis")
            win_rate_chart = create_win_rate_chart(csv_dict)
            if win_rate_chart:
                st.plotly_chart(win_rate_chart, use_container_width=True)
                
                # Additional win rate insights
                win_rate_data = get_dataframe_from_result(csv_dict, 'win_rates')
                if win_rate_data is not None and not win_rate_data.empty:
                    st.subheader("📊 Detailed Win Rate Statistics")
                    st.dataframe(win_rate_data, use_container_width=True)
            else:
                st.warning("No win rate data available")
        
        # Tab 4: Opening Analysis
        with tab4:
            st.subheader("♟️ Opening Performance Analysis")
            
            # Create opening charts
            white_count, black_count, white_perf, black_perf = create_opening_charts(csv_dict)
            
            # Display charts in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### As White")
                if white_count:
                    st.plotly_chart(white_count, use_container_width=True)
                else:
                    st.warning("No white opening count data available")
                
                if white_perf:
                    st.plotly_chart(white_perf, use_container_width=True)
                else:
                    st.warning("No white opening performance data available")
            
            with col2:
                st.markdown("### As Black")
                if black_count:
                    st.plotly_chart(black_count, use_container_width=True)
                else:
                    st.warning("No black opening count data available")
                
                if black_perf:
                    st.plotly_chart(black_perf, use_container_width=True)
                else:
                    st.warning("No black opening performance data available")
        
        # Tab 5: Raw Data
        with tab5:
            st.subheader("📋 Raw Data Export")
            
            # Show data tables using helper function
            for key in csv_dict.keys():
                if key == 'overall_stats':
                    # Special handling for overall stats
                    st.write(f"**Overall Statistics**")
                    st.json(csv_dict[key])
                    st.markdown("---")
                else:
                    df = get_dataframe_from_result(csv_dict, key)
                    if df is not None and len(df) > 0:
                        st.write(f"**{key.replace('_', ' ').title()}** ({len(df)} records)")
                        
                        # Show sample data
                        if len(df) > 10:
                            st.write("📋 Sample data (first 10 rows):")
                            st.dataframe(df.head(10), use_container_width=True)
                        else:
                            st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {key}.csv",
                            data=csv,
                            file_name=f"{username}_{key}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key=f"download_{key}"
                        )
                        st.markdown("---")
    else:
        st.info("👈 Please enter your Lichess username and click 'Load Data' to get started!")
        
        # Add some helpful information
        with st.expander("ℹ️ How to use this dashboard", expanded=True):
            st.markdown("""
            ### Getting Started:
            1. **Enter your Lichess username** in the sidebar
            2. **Select date range** for the games you want to analyze
            3. **Choose game variant** (blitz, bullet, rapid, etc.)
            4. **Apply filters** if needed (rated games, analyzed games)
            5. **Click 'Load Data'** to fetch and process your games
            
            ### Features:
            - **📈 Data Summary**: Overview of your chess performance
            - **📊 Elo Progression**: Rating changes over time with trend analysis
            - **🎯 Win Rates**: Performance statistics by color
            - **♟️ Opening Analysis**: Your most played openings and their success rates
            - **📋 Raw Data**: Download your processed data as CSV files            
            """)

if __name__ == "__main__":
    main()