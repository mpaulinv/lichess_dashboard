# 🏆 Lichess Chess Dashboard

A comprehensive **Streamlit-based dashboard** for analyzing your chess performance on Lichess. Visualize your rating progression, analyze opening repertoire, track win rates, and gain insights into your chess games with beautiful, interactive charts.

## ✨ Features

### 📊 **Data Analysis**
- **Elo Progression**: Track rating changes over time with polynomial trend analysis
- **Win Rate Analysis**: Performance breakdown by color (White/Black) 
- **Opening Performance**: Analyze your most played openings and success rates
- **Game Results Distribution**: Visual breakdown of wins, draws, and losses

### 🎯 **Interactive Visualizations**
- Real-time chart generation with Plotly
- Responsive design that works on all screen sizes
- Multiple chart types: line charts, bar charts, pie charts, and stacked bars
- Hover tooltips and interactive legends

### 🔍 **Advanced Filtering**
- **Date Range Selection**: Analyze specific time periods
- **Game Variants**: Blitz, Bullet, Rapid, Classical, Correspondence
- **Rated vs Casual**: Filter by game type
- **Computer Analysis**: Include only analyzed games

### 📈 **Performance Insights**
- Rating statistics (highest, lowest, net change)
- Color-based performance analysis  
- Opening repertoire strength assessment
- Game count and time period analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Lichess account with public games

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/lichess_dashboard.git
   cd lichess_dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   streamlit run scripts/dashboard_app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📖 How to Use

### Getting Started
1. **Enter your Lichess username** in the sidebar
2. **Select date range** for the games you want to analyze  
3. **Choose game variant** (blitz, bullet, rapid, etc.)
4. **Apply filters** if needed (rated games, analyzed games)
5. **Click 'Load Data'** to fetch and process your games

### Dashboard Sections

#### 📈 **Data Summary**
- Quick overview of total games, win rate, average rating
- Game results distribution pie chart
- Key performance metrics at a glance

#### 📊 **Elo Progression** 
- Interactive line chart showing rating changes over time
- Polynomial trend line for long-term progress analysis
- Statistics: highest rating, lowest rating, net change

#### 🎯 **Win Rates**
- Performance breakdown by playing color
- Stacked bar charts showing win/draw/loss percentages
- Detailed statistics table

#### ♟️ **Opening Analysis**
- Top 10 most played openings as White and Black
- Performance statistics for each opening
- Win/Draw/Loss breakdown by opening

#### 📋 **Raw Data**
- Download processed data as CSV files
- View sample data for each dataset
- Export capabilities for further analysis

## 🏗️ Project Structure

```
lichess_dashboard/
├── scripts/
│   ├── dashboard_app.py          # Main Streamlit application
│   ├── data_processing.py        # Data processing and analysis
│   ├── lichess_api.py           # Lichess API integration
│   ├── pgn_parser.py            # PGN parsing for opening data
│   ├── dashboard_app_agent.py   # AI-powered insights (experimental)
│   └── requirements.txt         # Python dependencies
├── requirements.txt             # Root dependencies
└── README.md                   # Project documentation
```

## 🔧 Technical Details

### Data Sources
- **Lichess API**: Fetches game data in NDJSON format
- **PGN Parsing**: Extracts opening information using python-chess
- **Real-time Processing**: Games are processed and cached for performance

### Technologies Used
- **Frontend**: Streamlit for web interface
- **Data Processing**: Pandas, NumPy for data manipulation
- **Visualizations**: Plotly for interactive charts
- **API Integration**: Requests for Lichess API calls
- **Chess Logic**: python-chess for PGN parsing

### Performance Optimizations
- Parallel processing for large datasets
- Efficient data caching mechanisms
- Optimized DataFrame operations
- Streamlined API calls with error handling
