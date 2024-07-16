# ShopSecuritySystem

## Overview
ShopSecuritySystem is a project developed as part of the DPL302m course. It uses YOLOv8 to detect people and knives in a retail shop environment, aiming to enhance shop security by providing real-time alerts for potential threats.

## Features: 
- People Counting: Counts the number of people entering and exiting the shop.
- Knife Detection: Detects the presence of knives and alerts shop personnel.
- Real-time Processing: Utilizes YOLOv8 for fast and accurate object detection.


## Demo:
![client](https://github.com/TruongSinhAI/ShopSecuritySystem/assets/115483496/eb7fdcf1-a3f4-41d3-8aef-4b5a7da5f961)

## Output:
![result2](https://github.com/TruongSinhAI/ShopSecuritySystem/assets/115483496/de899be2-cdf5-4106-98d7-bfdfa8c135da)

## Project Structure
- Count_People: Contains code for detecting and counting people.
- DetectKnife: Contains code for detecting knives.
- Data: Includes datasets used for training and testing.
- Model: Contains the trained YOLOv8 model and related scripts.



## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/TruongSinhAI/ShopSecuritySystem.git
   cd ShopSecuritySystem
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
Run the system with the following command:
   ```bash
   cd Application
   python app.py
   ```
