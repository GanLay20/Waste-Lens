# Waste-Lens
WasteLens is an AI-powered web app that provides instant classification of waste to help users sort their trash correctly. Just point your camera at an item, and the app will tell you if it's Organic or Recyclable.

This project was developed by Yan Myo Aung (Team Solo) at Hack the North 2025.

## The Problem

In Canada, we generate over 3 million tonnes of plastic waste every year, and less than 11% is recycled. A major cause is recycling contamination when non-recyclable items are put in the wrong bin, they can cause entire truckloads of good materials to be sent to a landfill.

This project aims to solve this problem by removing the primary cause: user confusion.

## About the Dataset

This project uses the Waste Classification Data dataset, a publicly available collection of images from Kaggle.

    Source: Kaggle - Waste Classification Data

    Content: The dataset contains over 22,500 images of waste, categorized into two main classes: Organic and Recyclable.

    Structure: The data is pre-split into TRAIN and TEST directories, with subfolders for each class, making it ideal for supervised learning tasks.

    Purpose: For this project, the dataset was used to train a ResNet-18 model to distinguish between the two primary types of waste, forming the core of the WasteLens classification engine.

## Features

    Real-Time Classification: Uses a live camera feed to provide instant feedback.

    Simple Binary Output: A clear, easy-to-understand result (Organic / Recyclable).

    Lightweight Model: Built with an efficient model to run smoothly in the browser.

## Built With

    Machine Learning: Python, PyTorch, ResNet-18

    Front-End: HTML/CSS and JS

    Back-End: FastAPI (for serving the model)


## Project Ouput

<img width="1698" height="1406" alt="Organic" src="https://github.com/user-attachments/assets/edb5550c-dd56-4b8a-92bb-15272cafd753" />
