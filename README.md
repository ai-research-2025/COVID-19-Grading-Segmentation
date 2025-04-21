
# COVID-19 Grading & Segmentation System

**Validation of COVID-19 Grading System based on Harris Hawks Optimization (HHO) and Variational Quantum Classifier using JLDS-2024**  
*Javaria Amin, Nadia Gul, Muhammad Sharif, Steven L. Fernandes* 
<a href='https://www.frontiersin.org/journals/radiology/articles/10.3389/fradi.2025.1457173/abstract'><img width='70px' src='https://www.faisafrica.com/wp-content/uploads/2020/11/frontiers-vector-logo.png'></a>

![Proposed System Architecture](system_architecture.png)

---

## Getting Started

### 1. Environment Setup
Git clone our repository and install the required dependencies
```bash
pip install tensorflow torch torchvision opencv-python pillow scikit-learn
```

### 2. Dataset Preparation
- **Classification Dataset**:  
  Download the [COVID-CT Dataset](https://www.dropbox.com/scl/fi/i4xntov2doyebjy9pos4j/dataset.zip?dl=0&e=1&rlkey=ld3vegd1ofnpupoupnia0nex2&st=nmw8a3zt) and organize into `data/classification/`.
- **Segmentation Dataset**:  
  Use the [COVID-19 CT Segmentation Dataset](http://medicalsegmentation.com/covid19/) and organize into `data/segmentation/images/` and `data/segmentation/masks/`.

### 3. Trained Models Weights
You can also download the trained models weights and use them directly
- Trained classification model [Download here](https://www.dropbox.com/scl/fi/t6b122vyhqah6l91owf5d/trained_model.pth?rlkey=i4t22i2cg311030opjd7lamw6&e=1&st=zh3vrya1&dl=0).
- Trained feature selector [Download here](https://www.dropbox.com/scl/fi/eh7xqt4m0dtiqktsdodos/feature_selector.pkl?rlkey=pou1jfqq0pw7uhlytb5uenwyh&e=1&st=fy8hqhof&dl=0).
- Trained U-Net weights: [Download here](https://www.dropbox.com/scl/fi/zmgdzjs68f3hcr8zo7kio/segmentation_model.h5?rlkey=3nxon1gwjp28iikj3wbz55m1t&e=1&st=465tz55n&dl=0).

---

## Training

### 1. Configuration

### 2. Run Pipeline
**Step 1: Volumetric Analysis & Grading**  
```bash
python volumetric_analysis.py
```

**Step 2: Classification Model Training**  
```bash
python train_classification.py
```
- Features extracted via ResNet-18 FC-1000 layer.
- HHO selects top 469 features.

**Step 3: Segmentation Model Training**  
```bash
python train_segmentation.py
```
- Uses U-Net with 47 layers.

---

## Evaluation

| Task          | Metric       | Performance   |
|---------------|--------------|---------------|
| Classification| Accuracy     | 99.0% (JLDS)  |
|               | ROC-AUC      | 0.98 ± 0.01   |
| Segmentation  | Dice Score   | 0.892         |

---

## Acknowledgements
- UCSD-AI4H/COVID-CT Dataset [[1]](https://arxiv.org/abs/2003.13865)
- COVID-19 CT Segmentation Dataset [[2]](http://medicalsegmentation.com/covid19/)
- Harris Hawks Optimization (HHO) algorithm [[3]](https://doi.org/10.1016/j.future.2019.02.028)

---

## Citation
```bibtex
@article{amin5validation,
  title={Validation of COVID-19 Grading System based on Harris Hawks Optimization (HHO) and Variational Quantum Classifier using JLDS-2024},
  author={Amin, Dr Javeria and Sharif, Muhammad and Fernandes, Steven and others},
  journal={Frontiers in Radiology},
  volume={5},
  pages={1457173},
  publisher={Frontiers}
}
```
