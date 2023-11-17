# LATAM: Software Engineer (ML & LLMs) Application Challenge
#### By "Eugenio Grytsenko" <yevgry@gmail.com>

## Overview

The Data Scientist, trained a model to predict the **probability of delay** for a flight taking off or landing at SCL airport. The model was trained with **public and real data**. There are 6 models provided by DS. Objectives:

- Determine the best model to run in **Production** environment.
- Trascribe the chosen one into challenge/model.py and test it.
- Implement the API with FastAPI using challenge/api.py and do stress tests.
- Deploy the API to Google Cloud Platform.
- Implement CI/CD by using Github Actions (workflows).

## Features

- Machine Learning model to predict the **probability of delay** for flights.
- An API endpint is provided to be able to run the model.

## Model Decision Analysis

Based on the classification reports from the Jupiter notebook, here's a description of each model's performance and efficiency:

| Model | Classifier | Precision 0 | Precision 1 | Recall 0 | Recall 1 | F1-Score 0 | F1-Score 1 | Accuracy |
|-|-|-|-|-|-|-|-|-|
| 4.b.i | XGB | 0.81 | 0.00 | 1.00 | 0.00 | 0.90 | 0.00 | 0.81 |
| 4.b.ii | LR | 0.82 | 0.56 | 0.99 | 0.03 | 0.90 | 0.06 | 0.81 |
| 6.b.i | XGB | 0.88 | 0.25 | 0.52 | 0.69 | 0.66 | 0.37 | 0.55 |
| 6.b.ii | XGB | 0.81 | 0.76 | 1.00 | 0.01 | 0.90 | 0.01 | 0.81 |
| 6.b.iii | LR | 0.88 | 0.25 | 0.52 | 0.69 | 0.65 | 0.36 | 0.55 |
| 6.b.iv | LR | 0.81 | 0.53 | 1.00 | 0.01 | 0.90 | 0.03 | 0.81 |

#### 4.b.i (XGBoost)

Overall Efficiency: Highly efficient for `class 0` with perfect recall and high precision. However, it completely fails to identify `class 1`, as indicated by zero precision and recall.

#### 4.b.ii (Logistic Regression)

Overall Efficiency: Shows good efficiency for `class 0` with high precision and very high recall. For `class 1`, it shows moderate precision but very low recall, indicating limited effectiveness in identifying `class 1` instances.

#### 6.b.i (XGBoost with Feature Importance and Balance)

Overall Efficiency: More balanced in identifying both classes. Shows high precision for `class 0` but moderate recall, and moderate precision for `class 1` with high recall. This model is better at identifying `class 1` instances compared to the previous ones.

#### 6.b.ii (XGBoost with Feature Importance, without Balance)

Overall Efficiency: Highly effective for `class 0` with perfect recall and high precision. For `class 1`, despite high precision, the recall is extremely low, indicating it rarely identifies `class 1` instances correctly.

#### 6.b.iii (Logistic Regression with Feature Importance and Balance)

Overall Efficiency: Similar to 6.b.i (XGBoost), this model shows a balanced performance. High precision for `class 0` but moderate recall, and moderate precision for `class 1` with high recall. This model demonstrates a good balance in identifying both classes.

#### 6.b.iv (Logistic Regression with Feature Importance, without Balance)

Overall Efficiency: Very efficient for `class 0` with perfect recall and high precision. However, like 6.b.ii (XGBoost), it has higher precision for `class 1` but very low recall, indicating a limited ability to identify `class 1` instances.

> In summary, models 6.b.i (XGBoost) and 6.b.iii (Logistic Regression) with feature importance and class balancing show a more balanced approach towards both classes, especially in identifying class 1 instances, which other models struggle with. The models without class balancing (4.b.i, 4.b.ii, 6.b.ii, 6.b.iv) are highly efficient for class 0 but have significant limitations in recognizing class 1.

## Conclusion

#### Data Science conclusions (according to DS from LATAM):

By looking at the results of the 6 trained models, it can be determined: there is no noticeable difference in results between _XGBoost_ and _LogisticRegression_. Does not decrease the performance of the model by reducing the features to the 10 most important. Improves the model’s performance when balancing classes, since it increases the recall of `class 1`.

#### My additional comments:

Based on the analysis, the choice for the most productive model should be between _6.b.i (XGBoost with Feature Importance and Balance)_ and _6.b.iii (Logistic Regression with Feature Importance and Balance)_, as both are trained with the top 10 features and class balancing. These models show a balanced performance in identifying both classes, especially `class 1`, which is a common challenge.

#### Decision criteria:

- If the priority is interpretability and computational efficiency, and if the dataset is not excessively large or complex, _6.b.iii (Logistic Regression with Feature Importance and Balance)_ would be the preferred choice.
- If the focus is on handling larger datasets and potentially achieving slightly better performance, especially in complex scenarios, _6.b.i (XGBoost with Feature Importance and Balance)_ would be more suitable.

Both models are well-suited for a balanced identification of both classes, with the choice depending on specific requirements such as computational resources, dataset size, and the need for interpretability.

#### Dive deeper into my analysis

Given that the dataset consists of real and public data, the choice between _6.b.i (XGBoost with Feature Importance and Balance)_ and _6.b.iii (Logistic Regression with Feature Importance and Balance)_ for a `Production` environment should consider the following factors:

- **Nature and Complexity of the Dataset**: if the dataset is large, complex, and possibly with non-linear relationships, _XGBoost (6.b.i)_ might be more suitable due to its ability to handle complexity and large data volumes efficiently.
- **Need for Interpretability**: in public data contexts, there might be a need for transparency and the ability to explain decisions made by the model. If interpretability is a key factor, _Logistic Regression (6.b.iii)_ would be preferable due to its simplicity and interpretability.
- **Computational Resources**: if there are constraints on computational resources, Logistic Regression tends to be less resource-intensive than XGBoost. In such cases, 6.b.iii (Logistic Regression) might be more practical.
- **Performance Metrics**: both models show balanced performance, but if the priority is slightly better performance in `class 1` identification, which might be crucial for real and public datasets, _XGBoost (6.b.i)_ could be a better choice.
- **Real-time or Batch Processing**: for real-time predictions, the computational efficiency of Logistic Regression could be beneficial. For batch processing, where time is less of a constraint, _XGBoost_ might be more suitable.

#### Recommendation for Production Environment:

- If the dataset is large and complex, and if computational resources and model performance (especially for `class 1`) are the priority over interpretability, _6.b.i (XGBoost with Feature Importance and Balance)_ would be more suitable.
- If interpretability, transparency, and computational efficiency are more critical, especially considering public scrutiny and understanding, _6.b.iii (Logistic Regression with Feature Importance and Balance)_ would be the preferred choice.

In summary, the decision hinges on the specific requirements and constraints of the production environment, as well as the priorities in terms of model performance, interpretability, and computational resource management.

For this challenge I would choose the model `6.b.i (XGBoost with Feature Importance and Balance)`! So, let's transcribe it to challenge/model.py as well and test it!

## Bugs Fixes and code improvements

#### File exploration.ipynb:

1. There were several bugs (Unexpected arguments) in `barplot` (Seaborn library) call. For example, this one:

```python
sns.barplot(flight_type_rate_values, flight_type_rate['Tasa (%)'])
```

According to the Seaborn `barplot` documentation `x` and `y` names of variables must by specified:

```python
sns.barplot(x=flight_type_rate_values, y=flight_type_rate['Tasa (%)'])
```
Reference at: https://seaborn.pydata.org/generated/seaborn.barplot.html

2. Simplified chained comparisons in `is_high_season` method:

From:
```python
    if ((fecha >= range1_min and fecha <= range1_max) or 
        (fecha >= range2_min and fecha <= range2_max) or 
        (fecha >= range3_min and fecha <= range3_max) or
        (fecha >= range4_min and fecha <= range4_max)):
        return 1
    else:
        return 0
```

To (improved):
```python
    if (range1_min <= target_date <= range1_max or 
        range2_min <= target_date <= range2_max or 
        range3_min <= target_date <= range3_max or
        range4_min <= target_date <= range4_max):
        return 1
    else:
        return 0
```

3. Removed redundant parenthesis and added simplified chained comparisons in `get_period_day` method:

From:
```python
    if(date_time > morning_min and date_time < morning_max):
        return 'mañana'
    elif(date_time > afternoon_min and date_time < afternoon_max):
        return 'tarde'
    elif(
        (date_time > evening_min and date_time < evening_max) or
        (date_time > night_min and date_time < night_max)
    ):
        return 'noche'
```

To (improved):
```python
    period_morning = 'mañana'
    period_afternoon = 'tarde'
    period_night = 'noche'

    if morning_min < date_time < morning_max:
        return period_morning
    elif afternoon_min < date_time < afternoon_max:
        return period_afternoon
    elif evening_min < date_time < evening_max:
        return period_night
    elif night_min < date_time < night_max:
        return period_night
```

NOTE: also, some variable names within methods were renamed to naming standards to English.

4. Model's code improvements changed the implementation of `get_min_diff` method from applying to simplifying it by converting with `pd.to_datetime` the entire column into a datetime format, and then the subtraction operation is performed directly on the columns:
- Old `get_min_diff` performance observations: 4 tests passed in 10.80s
- New `get_min_diff` performance observations: 4 tests passed in 1.91s

From:
```python
def get_min_diff(data):
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds())/60
    return min_diff
```

To (improved performance):
```python
def get_min_diff(target_data):
    date_o = pd.to_datetime(target_data['Fecha-O'])
    date_i = pd.to_datetime(target_data['Fecha-I'])
    min_diff = (date_o - date_i).dt.total_seconds() / 60
    return min_diff
```

6. Small bug in definition of the return type of `preprocess` method from

From (`Generics should be specified through square brackets`):
```python
...
    def preprocess(
            self,
            data: pd.DataFrame,
            target_column: str = None
    ) -> Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame):
              ^                                               ^
...
```

To (fixed):
```python
...
    def preprocess(
            self,
            data: pd.DataFrame,
            target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
...
```

## TODO

- Improve model training: review the architectures and hyperparameters.
- Provide more balanced datasets and more historical data.
- Make API endpoints more secure.
- Implement load balancing, HA and fault tolerance.
- Contingency and disaster recovery.

## Tech

This challenge uses a number of open source projects to work properly:

- [Python] - high-level programming language
- [ML] - machine learning (Sklearn and XGBoost)
- [Dataframe] - using Numpy and Pandas
- [API] - application interface for ML endpoints (FastAPI)
- [Docker] - containers for API deployments
- [Cloud] - deployments to Google Cloud Platform
- [Visualization] - data analysis graphs (Matplotlib and Seaborn)
- [PyCharm] - awesome integrated development environment for my code writing ;-)
- [Jupiter] - jupiter notebooks integrated in PyCharm

## Run

The challenge requires [Python](https://www.python.org/downloads/) v3+ to run.

Install the dependencies and start the server.

```sh
make install
```

Run all project tests.

```sh
make model-test
make api-test
make stress-test
```

Make sure you've configured properly the `STRESS_URL` within Makefile (line 26).

```sh
STRESS_URL = http://127.0.0.1:8000
```

## API & Continuous Integration / Continuous Delivery

In order to implement CI/CD and to be able to deploy the API, I've configured GitHub Actions, a feature of GitHub that allows you to automate, customize, and execute the software development workflows right in the repository.

This challenge will be deployed to Google Cloud Platform. Files to consider:

| Path | Description |
|-|-|
| .github/workflows/ci.yml | Continuous Integration rules |
| .github/workflows/cd.yml | Continuous Delivery rules |
| Dockerfile | A recipe to accomplish delivery to GCP environment |

## License

This challenge is under NDA of LATAM and DataArt Solutions Inc., and its associates.
Signed by: Eugenio Grytsenko (Date: 10/30/23)
