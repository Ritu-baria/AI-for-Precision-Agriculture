# 🌾 AI for Precision Agriculture

This project leverages Artificial Intelligence and Data Science to transform traditional farming into data-driven precision agriculture. By using weather data, soil health metrics, satellite imagery, and computer vision, this system can predict crop yield, monitor crop health, detect leaf diseases, and recommend optimal farming practices.

---

## 📌 Project Objectives

- 🧠 Predict crop yield based on soil, climate, and crop type
- 🛰 Analyze vegetation health using NDVI (satellite data)
- 🧪 Detect crop diseases using CNN on leaf images
- 💧 Recommend irrigation/fertilizer strategies to optimize yield and reduce waste

---

## 📊 Dataset Sources

| Data Type        | Source                                                                            |
|------------------|-----------------------------------------------------------------------------------|
| Crop Yield       | [data.gov.in](https://data.gov.in)                                                |
| Weather Data     | [NASA POWER](https://power.larc.nasa.gov/)                                       |
| Soil Data        | [ICRISAT India Soil Database](https://www.icrisat.org/)                          |
| Satellite Imagery| [Sentinel-2 (NDVI)](https://developers.google.com/earth-engine/datasets)         |
| Leaf Disease     | [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## 🧠 Technologies Used

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn (RandomForest, XGBoost)
- TensorFlow / Keras (for CNN model)
- Rasterio, OpenCV, GeoPandas (for satellite data analysis)
- Streamlit (optional dashboard deployment)

---

## 📁 Folder Structure

```bash
.
├── data/                             # CSVs and satellite image samples
├── notebooks/
│   ├── 1_crop_yield_prediction.ipynb
│   ├── 2_ndvi_analysis.ipynb
│   ├── 3_leaf_disease_cnn.ipynb
├── leaf_disease_model.h5            # Trained CNN model
├── crop_yield_model.pkl             # Saved ML model for yield prediction
├── app/                             # Streamlit dashboard files
├── requirements.txt
└── README.md
# AI-for-Precision-Agriculture
