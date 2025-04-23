import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import make_scorer


class KNNOptimizer:
    """
    A class for optimizing KNN models with focus on various metrics like specificity,
    sensitivity, and accuracy.
    """

    def __init__(self):
        self.best_specificity = None
        self.param_grid = None
        self.best_sensitivity = None
        self.best_accuracy = None
        self.best_k_index = None
        self.param_values = None
        self.best_k = None
        self.best_score = None
        self.best_model = None
        self.cv_results = None
        self.grid_search = None
        self.all_scores = None
        self.metric_values = {}

    def optimize(self, X_train, y_train, k_range=range(1, 31), cv=None,
                 optimization_metric='specificity', calculate_accuracy=None,
                 calculate_specificity=None, calculate_sensitivity=None):
        """
        Optimize KNN model by finding the best k value for a given metric.
        """
        # Convert k_range to a list explicitly to avoid ufunc issues
        self.param_grid = {'n_neighbors': list(k_range)}

        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(calculate_accuracy),
            'specificity': make_scorer(calculate_specificity),
            'sensitivity': make_scorer(calculate_sensitivity)
        }

        # Create base KNN model
        knn = KNeighborsClassifier()

        # Use GridSearchCV to find the best k value
        self.grid_search = GridSearchCV(
            estimator=knn,
            param_grid=self.param_grid,
            cv=cv,
            scoring=scoring,
            refit=optimization_metric,
            return_train_score=True
        )

        # Run grid search
        self.grid_search.fit(X_train, y_train)

        # Get best parameters and score
        self.best_k = self.grid_search.best_params_['n_neighbors']
        self.best_specificity = self.grid_search.best_score_
        self.best_score = self.grid_search.best_score_

        # Get all results
        self.all_scores = self.grid_search.cv_results_

        self.param_values = self.all_scores['param_n_neighbors']
        self.best_k_index = np.where(self.all_scores['param_n_neighbors'].data == self.best_k)[0][0]
        self.best_accuracy = self.all_scores['mean_test_accuracy'][self.best_k_index]
        self.best_sensitivity = self.all_scores['mean_test_sensitivity'][self.best_k_index]

        # Create best model
        self.best_model = KNeighborsClassifier(n_neighbors=self.best_k)

        self.metric_values = {
            'accuracy': float(self.best_accuracy),
            'specificity': float(self.best_specificity),
            'sensitivity': float(self.best_sensitivity)
        }

        return self

    def plot_metrics(self):
        """
        Plot metrics vs k values.
        """
        if self.all_scores is None:
            raise ValueError("You must run optimize() before plotting metrics.")

        fig = plt.figure(figsize=(12, 8))

        # Plot della specificità
        plt.plot(self.param_grid['n_neighbors'], self.all_scores['mean_test_specificity'],
                 marker='o', linestyle='-', color='blue', label='Specificità')

        # Plot dell'accuracy
        plt.plot(self.param_grid['n_neighbors'], self.all_scores['mean_test_accuracy'],
                 marker='s', linestyle='--', color='green', label='Accuracy')

        # Plot della sensibilità
        plt.plot(self.param_grid['n_neighbors'], self.all_scores['mean_test_sensitivity'],
                 marker='^', linestyle='-.', color='red', label='Sensibilità')

        # Evidenzia il miglior k per specificità
        plt.axvline(x=self.best_k, color='gray', linestyle=':', alpha=0.7)
        plt.scatter([self.best_k], [self.best_specificity], s=200, facecolors='none', edgecolors='blue', linewidth=2)

        plt.title(f'Metriche vs. k (miglior k={self.best_k}, specificità={self.best_specificity:.4f})')
        plt.xlabel('Numero di vicini (k)')
        plt.ylabel('Punteggio medio (CV)')
        plt.xticks(self.param_grid['n_neighbors'][::2])  # Mostra solo un sottoinsieme dei valori k per chiarezza
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return fig

    def print_results(self):
        """
        Print optimization results.
        """
        if self.best_k is None:
            raise ValueError("You must run optimize() before printing results.")

        # Get the optimization metric name
        best_metric_name = self.grid_search.refit

        print(f'Miglior valore di k: {self.best_k}, '
              f'Miglior {best_metric_name}: {self.best_score:.4f}')

        print(f'Con k = {self.best_k}: '
              f'Accuracy = {self.metric_values.get("accuracy", 0.0):.4f}, '
              f'Specificità = {self.metric_values.get("specificity", 0.0):.4f}, '
              f'Sensibilità = {self.metric_values.get("sensitivity", 0.0):.4f}')

        print(f'Come suggerisce il grafico, il miglior risultato per '
              f'{best_metric_name} si ottiene con k = {self.best_k}.')

    def cross_validate_best_model(self, X_train, y_train, cv=5,
                                  calculate_accuracy=None, calculate_specificity=None,
                                  calculate_sensitivity=None):
        """
        Cross-validate the best model.
        """
        if self.best_model is None:
            raise ValueError("You must run optimize() before cross-validating.")

        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(calculate_accuracy),
            'specificity': make_scorer(calculate_specificity),
            'sensitivity': make_scorer(calculate_sensitivity)
        }

        # Cross-validate best model
        self.cv_results = cross_validate(self.best_model, X_train, y_train,
                                         cv=cv, scoring=scoring)

        return self.cv_results

    def print_cv_results(self):
        """
        Print cross-validation results.
        """
        if self.cv_results is None:
            raise ValueError("You must run cross_validate_best_model() before printing CV results.")

        # Get the optimization metric name
        best_metric_name = self.grid_search.refit

        print(f"\nMetriche del modello ottimizzato per {best_metric_name}:")
        print("Accuracy:", float(np.mean(self.cv_results['test_accuracy'])))
        print("Specificità:", float(np.mean(self.cv_results['test_specificity'])))
        print("Sensibilità:", float(np.mean(self.cv_results['test_sensitivity'])))

    def get_results_for_dataframe(self, type_data=None, model_name=None):
        """
        Get results formatted for adding to a DataFrame.
        """
        if self.cv_results is None:
            raise ValueError("You must run cross_validate_best_model() before getting DataFrame results.")

        # Get the optimization metric name
        best_metric_name = self.grid_search.refit

        if model_name is None:
            model_name = f"KNN {type_data} (ottimizzato per {best_metric_name})"

        return [
            model_name,
            float(np.mean(self.cv_results['test_accuracy'])),
            float(np.mean(self.cv_results['test_specificity'])),
            float(np.mean(self.cv_results['test_sensitivity']))
        ]

    def fit(self, X_train, y_train):
        """
        Fit the best model to the training data.
        """
        if self.best_model is None:
            raise ValueError("You must run optimize() before fitting.")

        self.best_model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """
        Make predictions using the best model.
        """
        if self.best_model is None or not hasattr(self.best_model, 'classes_'):
            raise ValueError("You must fit the model before predicting.")

        return self.best_model.predict(X)