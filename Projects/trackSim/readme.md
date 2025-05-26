# 🏁 Lap Time Simulator with Tire Degradation, Strategy, and Traffic Effects

This project simulates lap-by-lap performance of race cars on a circuit using Python. Inspired by real-world F1 vehicle dynamics, it models tire degradation, pit strategy, fuel burn, and overtaking logic to evaluate and compare race outcomes.

## 🚀 Features

- **Track-Sector Simulation**: Straights and corners with unique grip, radius, and sector type
- **Tire Degradation**: Compound-specific wear rates, thermal effects, and nonlinear decay
- **Fuel Burn Modeling**: Reduces car mass each lap to affect acceleration and braking
- **Traffic & Overtake Logic**: Position-switching if time delta exceeds thresholds
- **Pit Strategy Engine**: Supports lap-based and grip-threshold-based pit stops
- **Multi-Car Comparison**: Head-to-head performance simulation with visualized outcomes
- **Visuals**: Sector heatmaps, lap time graphs, and compound transitions

## 📂 File Structure

| File | Description |
|------|-------------|
| `LapTimeSimulator.ipynb` | Google Colab notebook containing simulation code and visualizations |
| `circuit.csv` | Example track configuration for CSV import |
| `README.md` | This file |
| `images/` | Folder for heatmaps, lap plots, and compound comparison visuals |

## 🛠️ Dependencies

Run in [Google Colab](https://colab.research.google.com/) or locally with:

```bash
pip install numpy pandas matplotlib seaborn
