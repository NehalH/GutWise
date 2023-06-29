# GutWise
A Logistic Regression classification model for determining gut health.

## About
GutWise is an innovative project focused on promoting gut health and providing valuable insights into the intricate relationship between lifestyle factors and the gut microbiota.

This repository serves as a central hub for all the code, resources, and documentation related to GutWise. Our aim is to develop a comprehensive platform that empowers individuals to make informed choices for improving their gut health and overall well-being.

Inside this repository, you will find various components that make up the GutWise system. This includes the classification models and algorithms used to analyze gut health data, data preprocessing scripts, visualization tools, and other supporting resources.

We encourage collaboration and contributions from the open-source community to further enhance the capabilities of GutWise. Whether you are an expert in data science, machine learning, or have a keen interest in gut health, there are ample opportunities to get involved and make a meaningful impact.

By leveraging the collective expertise and knowledge of contributors like you, we can continue to advance research and provide valuable tools and insights to promote optimal gut health. Together, we can create a healthier future, one gut at a time.

Thank you for your interest in GutWise. I invite you to explore this repository and contribute to this exciting and impactful project.

Note: The GutWise project is not intended to replace professional medical advice or diagnosis. It is designed to provide information and tools for individuals interested in gut health and should be used in conjunction with guidance from healthcare professionals.


## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/NehalH/GutWise.git
   ```
2. Navigate to the project directory:

   ```shell
   cd GutWise
   ```
3. Install the required dependencies:
   ```shell
   pip install -r requirements.txt
   ```

## Custom usage

1. Define and strip dataset:
   
    Remove any unwanted columns or rows from your .csv file (dataset) and save it as ```dataset.csv```.

3. Place the dataset:
   
    Place the .csv file in ./src/dataset/ directory.
   
4. Define class header:
   
    Change the value of the variable ```target_variable``` to the column header of your class column.

6. Preprocess the dataset:
   
     Run the following command in the terminal:
     ```
     python ./src/preprocessor.py
     ```
7. Train and test:

     Run the following command in the terminal:
     ```
     python ./src/main.py
     ```
