1. Title: Smartphone Dataset for Crowd Anomaly Detection

2. Relevant Information:
   -- This dataset is collected by conducting two experiments at different times and different participants.
      We collected more dataset to improve the accuracy of our anomaly detection algorithms applied in a crowd.
      The dataset was collected from the in-built accelerometer and gyroscope of a smartphone placed inside the pockets of participants.

      The data was collected from 8 participants within the age group of 22-42 years from experiment 1, and 15 participants from experiment 2. 
      The anomaly based activity was performed for 1sec multiple times and the 3-axial linear acceleration and 3-axial angular velocity  were 
      collected at a constant rate of 100Hz.
   -- These results are presented in a paper titled: 'Anomaly Detection in Crowds using Multi Sensory Information' has been 
       published in 5th IEEE International Conferenceon Advanced Video and Signal-based Surveillance (AVSS).

2. Source Information
   -- Creators: Muhammad Irfan
   -- School of Electronic Engineering and Computer Science, Queen Mary University of London, UK
      Donors: Muhammad Irfan (mirfan83@yahoo.com)
   -- Date: July, 2020
 
3. Usage:
   --Results: We obtained an overall accuracy of 93% with Random Forest algorithm after balancing the two classes.
              The accuracies were obtained through a 10-fold cross-validation method.

5. Number of Instances: 
   -- Experiment 1 = 1727
   -- Experiment 2 = 1917

6. How to open the dataset files
   -- There are four files included in the repository. Two files for each experiment, one with original number of samples and one with balanced data.
   -- Convert the extension for each file to ARFF from TXT to open in Weka software before running.
   -- We use Weka Machine Learning tool to open, analyse, train and test the data. Weka is an open source tool freely available
      to download from the link given below:
   -- https://www.cs.waikato.ac.nz/ml/weka/

7. Number of Attributes: 24 (See features.txt and features_info.txt included)