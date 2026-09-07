# Task-03-Tayyba-Yahya
# AI Tech Stack Recommender

## Project 3 - AI Recommendation Logic

This project is a simple AI-based recommendation system that recommends career paths based on the skills and interests entered by the user.

The system uses **Content-Based Filtering** with **TF-IDF** and **Cosine Similarity** to compare the user's skills with the skills required for different career paths.

## Features

* Takes multiple skills or interests from the user
* Requires at least 3 inputs
* Cleans and normalises skill names
* Handles different names for the same skill
* Uses TF-IDF to convert skills into numerical vectors
* Uses Cosine Similarity to calculate matching scores
* Sorts career paths according to their similarity score
* Displays the Top 3 recommended career paths
* Shows which skills the user already has
* Handles unknown skills using a popularity-based fallback
* Includes a demo mode for testing different profiles

## How It Works

The recommendation process follows these steps:

```text
User Input
    ↓
Skill Normalisation
    ↓
TF-IDF Vectorisation
    ↓
Cosine Similarity
    ↓
Sorting by Match Score
    ↓
Top 3 Career Recommendations
```

### 1. User Input

The user enters at least three skills or interests.

Example:

```text
Python
Cloud Computing
Automation
```

### 2. Skill Normalisation

The system converts skills to lowercase and handles common alternative names.

For example:

```text
AI → Machine Learning
ML → Machine Learning
JS → JavaScript
Py → Python
Cloud → Cloud Computing
```

### 3. TF-IDF

TF-IDF is used to convert the skills into numerical vectors. This helps the system represent the importance of different skills.

### 4. Cosine Similarity

The user's skill profile is compared with each career path using Cosine Similarity.

A higher similarity score means that the career path has a better match with the user's skills.

### 5. Ranking

The career paths are sorted from the highest similarity score to the lowest score.

The system then displays the Top 3 recommendations.

## Dataset

The project uses `raw_skills.csv` as the dataset.

The dataset contains:

* Career role
* Required skills
* Popularity value

The CSV file is used by the Python program to create the career profiles for recommendation.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Cosine Similarity

## Project Structure

```text
tech-stack-recommender/
│
├── tech_stack_recommender.py
├── raw_skills.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Make sure Python is installed on your computer.

Clone the repository:

```bash
git clone <your-github-repository-link>
```

Move into the project folder:

```bash
cd tech-stack-recommender
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

Run the main program:

```bash
python tech_stack_recommender.py
```

The program will ask you to enter your skills.

You need to enter at least three skills.

## Demo Mode

The project also includes a demo mode with sample profiles.

Run:

```bash
python tech_stack_recommender.py --demo
```

The demo tests different types of profiles, including technical skills and unknown skills.

## Example

Input:

```text
Python
Cloud Computing
Automation
```

The system compares these skills with the career paths in the dataset and displays the best matching career paths.

Example output:

```text
TOP 3 RECOMMENDED CAREER PATHS

1. Cloud Engineer   match: 75.2%
   Skills: Python, AWS, Cloud Computing

2. DevOps Engineer  match: 63.8%
   Skills: Linux, Cloud Computing, Automation

3. Python Developer match: 51.4%
   Skills: Python, Automation, Git
```

## Cold Start Handling

If none of the skills entered by the user are available in the dataset, the system uses the popularity value to show the most popular career paths instead of returning meaningless similarity scores.

## Learning Outcome

This project helped demonstrate basic recommendation-system concepts such as:

* Data ingestion
* Text preprocessing
* Feature extraction
* TF-IDF
* Cosine Similarity
* Ranking
* Content-based filtering
* Handling unknown user inputs

## Author

**[Tayyba Yahya]**

DecodeLabs Industrial Training Kit - Batch 2026

## License

This project was created for educational and training purposes.
