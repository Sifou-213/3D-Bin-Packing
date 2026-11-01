# 3D Bin Packing · Truck loading solver

**▶ Live solver: https://colis-zj84.onrender.com/**
(free hosting: the first load can take about a minute while the server wakes up)

Final-year project (Master in Operations Research, 2025) carried out at Condor Logistics, Algeria.

## Problem

How should each truck be loaded to lower the transport cost per item and avoid damage caused by non-compliant loads?

## Approach

- Modelled as a **3D Bin Packing / container loading problem**.
- Solved with a **hybrid genetic algorithm** that builds the load **layer by layer**, from the back of the truck to the door.
- Real constraints included: catalogue of the **11 vehicles** of the fleet, **allowed orientations** for each box, computation time limit.

## Application

- Python web app (**Flask**) with a **REST API** (`POST /solve`).
- Computes the load plan and displays it in **3D**, layer by layer (Plotly), with fill metrics.
- Adjustable algorithm parameters: population size, elitism, merge mutations, max generations, time limit.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000.

## Author

Sif Eddine Benabida · Portfolio: https://sifou-213.github.io · GitHub: https://github.com/Sifou-213
