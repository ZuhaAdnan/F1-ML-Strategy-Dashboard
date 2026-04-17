# Machine Learning Driven Formula 1 Race Strategy Dashboard

## Overview
This project applies machine learning to Formula 1 race strategy using historical race data from the 2023 Italian Grand Prix (Monza). It predicts key race decisions including pit strategy, lap times, overtaking probability, and post pit position.

The system is implemented as a predictive dashboard using trained machine learning models.

---

## Objectives
- Predict future lap times using regression models
- Classify pit stop decisions
- Estimate overtaking probability
- Predict post pit rejoin position
- Provide real time race strategy support

---

## Dataset
- Source: FastF1 Python library
- Race: 2023 Italian Grand Prix (Monza)
- Data includes:
  - Lap times
  - Tyre compounds
  - Driver information
  - Position data

---

## Features Used
- Tyre age
- Encoded driver ID
- Encoded tyre compound
- Position change
- Next race position

---

## Models Used

### Lap Time Prediction
- Model: Random Forest Regressor
- Predicts lap time in seconds based on tyre and driver data

### Pit Stop Decision Model
- Model: Random Forest Classifier
- Outputs: Stay Out, Pit Soon, Pit Now

### Overtake Probability Model
- Binary classification model
- Predicts probability of position gain

### Post Pit Position Prediction
- Regression model
- Predicts expected position after pit stop

---

## System Design
- Offline model training using historical data
- Real time inference using dashboard interface
- Interactive lap based simulation

---

## Results
- Model successfully captures tyre degradation trends
- Pit stop model reacts to increasing tyre age
- Overtake probability increases in mid race phases
- Post pit position prediction aligns with race dynamics

---

## Limitations
- No weather data included
- Safety car effects not modeled
- Traffic and race incidents not included

---

## Future Improvements
- Include weather and track condition data
- Improve model accuracy with deeper learning models
- Add reinforcement learning for strategy optimization

---

