# 128-Honors-Project
Prabhat Kalle (pskalle2) and Shubham Mital (smital2)

**Introduction**
  * This project focuses on developing a house price prediction model using Rust. We aim to implement and compare different machine learning algorithms to predict house prices based on various characteristics like location, size, and number of bedrooms. The motivation behind selecting this project is to explore the capabilities of Rust in the field of machine learning and data processing. Rust's performance and safety features make it an interesting choice for implementing efficient and reliable predictive models.

**Technical Overview**
  * Data Collection: Responsible for gathering and performing initial preprocessing on housing data from public datasets
  * Data Processing: Dedicated to cleaning, normalizing, and preparing the data for model training. This includes dealing with missing values, selecting relevant features, and splitting the data into training, validation, and testing sets
  * Machine Learning: Involves either leveraging existing Rust crates for machine learning or implementing models from scratch, followed by training these models with processed data and evaluating their performance.
  * API for Predictions: A Rust-built web server that allows users to input features of a house and receive price predictions, serving the trained model.

**Challenges**
  * The potential scarcity of comprehensive machine learning libraries in Rust, which might necessitate creating algorithms from scratch or interfacing with Python libraries for broader functionality.
  * Ensuring optimized performance for data processing and model training within the Rust ecosystem

