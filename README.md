# 🏭 Continuous Soil Remediation Facility Optimizer

A Streamlit web application that determines optimal treatment cell configurations for continuous-flow soil remediation facilities.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_APP_URL_HERE)

## 🎯 What It Does

This tool helps environmental engineers and facility planners design soil remediation facilities by answering: **"How many treatment cells of what size do I need to continuously process incoming soil?"**

### Key Features

- 📊 **Smart Configuration Optimization** - Tests hundreds of cell size/quantity combinations
- 🎯 **Balanced Recommendations** - Optimizes for capital cost, utilization, and operational efficiency
- 📈 **Interactive Visualizations** - Compare configurations and understand tradeoffs
- 💾 **Excel Reports** - Export detailed analysis and recommendations
- ⚙️ **Flexible Parameters** - Customize equipment capacities, phase durations, and weekend schedules

## 🚀 Quick Start

### Try It Online
Visit the live app: [YOUR_APP_URL_HERE]

### Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/soil-facility-optimizer.git
cd soil-facility-optimizer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run continuous_soil_facility_optimizer.py
```

Open http://localhost:8501 in your browser.

## 📖 How to Use

1. **Set Daily Volume** - Enter average daily incoming soil (e.g., 200 CY/day)
2. **Define Cell Depth** - Choose preferred treatment cell depth (e.g., 4 feet)
3. **Configure Equipment** - Set loading/unloading capacities
4. **Set Treatment Phases** - Define duration for Rip, Treat, and Dry phases
5. **Choose Weekend Schedule** - Enable/disable operations by phase and day
6. **Run Optimization** - Get instant recommendations!

## 🔍 What You Get

### Optimal Configuration
- Cell size (volume in CY)
- Number of cells needed
- Dimensions (Length × Width × Depth)
- Expected utilization rate

### Performance Metrics
- Total facility capacity
- Daily throughput
- Days of storage capacity
- Capacity surplus/deficit

### Alternative Options
- Top 10 ranked configurations
- Tradeoff analysis
- Visual comparisons

## 🎓 Example Use Cases

**Scenario 1: New Facility Design**
> "We're processing 250 CY/day. How many cells should we build?"

**Scenario 2: Capacity Expansion**
> "Our volume doubled. Do we add more cells or build larger ones?"

**Scenario 3: Budget Planning**
> "What's the minimum viable configuration, and what's optimal?"

## 📊 Key Metrics Explained

### Utilization
- **Target**: 80-90% for optimal balance
- **Below 70%**: Over-capacity (wasting capital)
- **Above 95%**: Too tight (risky for disruptions)

### Days of Capacity
How many days of incoming volume your facility can hold. Higher = more buffer for surges.

### Cycle Time
Total time for soil to go through: Load → Rip → Treat → Dry → Unload

## 🛠️ Technology Stack

- **Streamlit** - Web interface
- **Pandas** - Data manipulation
- **Plotly** - Interactive visualizations
- **OpenPyXL** - Excel report generation
- **NumPy** - Numerical computations

## 📦 Files Included

```
soil-facility-optimizer/
├── continuous_soil_facility_optimizer.py  # Main application
├── requirements.txt                       # Python dependencies
├── README.md                             # This file
├── DEPLOYMENT_GUIDE.md                   # Detailed deployment instructions
├── QUICK_START.md                        # 5-minute deployment guide
└── .gitignore                            # Git ignore rules
```

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Share feedback

## 📄 License

This project is provided as-is for environmental engineering applications.

## 🆘 Support

- **Issues**: Open a GitHub issue
- **Questions**: See DEPLOYMENT_GUIDE.md
- **Streamlit Help**: https://discuss.streamlit.io

## 🙏 Acknowledgments

Built with Streamlit Community Cloud - Free hosting for data apps!

## 📈 Version History

### v1.0.0 (Current)
- Initial release
- Core optimization engine
- Excel export functionality
- Interactive visualizations

---

**Made for environmental engineers, by engineers** 🌱

[⭐ Star this repo](https://github.com/YOUR_USERNAME/soil-facility-optimizer) if you find it useful!
