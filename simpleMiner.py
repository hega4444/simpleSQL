import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    # Disables excessive log
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # Disables ODNN

from itertools import combinations
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from stringcolor import *
import matplotlib
matplotlib.use("Agg")

#CONSTANTS
LINEAR_MODEL = 1
ANN_MODEL = 2


class SimpleMinerBaseObject():

    def __init__(self, db_object, table_name, color = None) -> None:
        
        self.db = db_object
        self.table_name = table_name
        self.model = None
        self.model_type = None
        self.columns = None
        self.y_test = None
        self.y_pred = None
        self.data_frame = None
        self.convert = None
        self.field_scores = None
        self.r2_limits = [0.8, 0.4, 0.0]
        self.r2_limits_c = [0.8, 0.4, 0.0]
        self.max_r2 = 0
        self.plot_limit = 3
        self.current_features = []
        self.current_target = None
        self.theme_color = 'Gold' if color is None else color
        self.one_hot_columns = []
        self.hide_unfit = True
        self.show_details = False
        self.mae = None
        self.mse = None
        self.rmse = None
        self.r2 = None

        self.table_fields = self.db.get_table_fields(table_name = self.table_name)

        if self.db.check_table_exists(self.table_name) is False:
            raise ValueError(f"Table '{table_name}' does not exist.")
        

    def verify_features_in_table(self, features):
        return all([True if f.capitalize() in self.table_fields else False for f in features])

    def get_data_frame(self, features, target):
             
            fields = ", ".join(features).lower() + ", " + target[0].lower() 
            query = f"SELECT {fields} FROM {self.table_name};"  
            data = list(self.db.into_dict(sql_block = query).values())
            
            self.data_frame = pd.DataFrame(data)
    
    def one_hot_encode_new_data(self, input_data):
        # Ensure the new data has the same columns as the one-hot encoded training data
        pass

    def define_model_features(self, features, target = None):

        if self.verify_features_in_table(features = features) is False:
            raise ValueError("Some features do not exist. Check table fields.")
        
        if target is None:
            if self.current_target:
                target = [self.current_target]
            else:
                raise ValueError("Missing target.")
        
        try:
            self.get_data_frame(features, target)
        except Exception:
            print("simpleMiner -Problems retrieving data.")
            return

        features = [f.capitalize() for f in features]
        self.current_features = features

    def predict(self, input_values):
        pass

    def show_model_performance(self):

        if not self.model:
            print("Model has not been initialized yet.")
        else:
            print("Model performance indicators:\n")

            print(f"Number of records in DataFrame: {len(self.data_frame)}")
            print(f"Mean Absolute Error (MAE): {self.mae}")
            print(f"Mean Squared Error (MSE): {self.mse}")
            print(f"Root Mean Squared Error (RMSE): {self.rmse}")
            print(f"R-squared (R2): {self.r2}")

    def set_custom_r2_limits(self, high = None, moderate = None, low = None, ranking= None):
        r2_1, r2_2, r2_3 = 0, 1, 2
        if high:
            self.r2_limits[r2_1] = high
        if moderate:
            self.r2_limits[r2_2] = moderate
        if low:
            self.r2_limits[r2_3] = low
        if ranking:
            self.plot_limit = ranking
        
    def reset_limits(self):
        self.r2_limits = self.r2_limits_c

    def analize_field_relations(self, target, hide_unfit = True, show_details = True):

        fields = self.db.get_table_fields(self.table_name)
        fields.remove(target.capitalize())
        target = target.capitalize()
        self.current_target = target
        try:
            fields.remove('Id')
        except Exception:
            pass

        self.field_scores = {}

        # Generate subgroups of different lengths
        subgroups = []
        for r in range(1, len(fields) + 1):
            subgroups.extend(combinations(fields, r))

        # Convert the subgroups to lists
        sub_features = [list(subgroup) for subgroup in subgroups]

        print(f"\nsimpleMiner - Analizing relations to field '{target}'. This may take some time...")

        # Display the subgroups
        max_r2 = 0
        for i, features in enumerate(sub_features):

            self.field_scores[i] = {'features': features}
            
            #Set new features for the model
            self.define_model_features(features=features, target=[target])

            self.field_scores[i]['scores'] = {}
            self.field_scores[i]['scores']['MAE'] = self.mae
            self.field_scores[i]['scores']['MSE'] = self.mse
            self.field_scores[i]['scores']['RMSE'] = self.rmse
            self.field_scores[i]['scores']['R2'] = self.r2

            if self.r2 > max_r2:
                max_r2 = self.r2
        
        self.max_r2 = max_r2

        rows = self.field_scores.values()

        table_data = []
        for r2 in rows:
            table_row = []
            table_row.append(", ".join(r2['features']))
            table_row.append(round(r2['scores']['MAE'], 3))
            table_row.append(round(r2['scores']['MSE'], 3))
            table_row.append(round(r2['scores']['RMSE'], 3))
            table_row.append(round(r2['scores']['R2'], 3))

            if hide_unfit:
                if (r2['scores']['R2'] >= 0 and r2['scores']['R2'] <= 1):
                    table_data.append(table_row)
            else:
                table_data.append(table_row)

        sorted_data = sorted(table_data, key=lambda x: (-x[4], x[1]))

        category = [''] *4
        new_cat = 0
        last = 1
        create = True
        r2_1, r2_2, r2_3 = 0, 1, 2

        for row in sorted_data:

            r2 = row[4]
   
            if create and new_cat == 0:
                color = self.theme_color if not show_details else 'Lime2'
                table = self.db.new_table(options= {'align':'l', 'color':f'{color}'})
                add_title = "" if not show_details else f" with high inference (R²>{self.r2_limits[r2_1]}) :"
                table.field_names = [f'Features{add_title}', 'MAE', 'MSE', 'RMSE', 'R²']
                create = False

            if show_details is True:

                if new_cat==0 and r2 >= self.r2_limits[r2_1]:
                    new_cat = 1

                elif new_cat > 0 and r2 >= self.r2_limits[r2_2] and r2 < self.r2_limits[r2_1]:
                    category[1] = f" with moderate chances (R²>{self.r2_limits[r2_2]})    :"
                    new_cat = 2

                elif new_cat > 0 and r2 > self.r2_limits[r2_3] and r2 < self.r2_limits[r2_2]:
                    category[2] = f" with low chances (R²>{self.r2_limits[r2_3]})         :"
                    new_cat = 3
                
                elif new_cat > 0 and r2 <= self.r2_limits[r2_3]:
                    category[3] = " with no inference (R²<0.0) :"
                    new_cat = 4

                if new_cat > last:
                    print()
                    print(table)

                    color = self.theme_color if not show_details else ['Lime2', 'Yellow2', 'Gold2', 'OrangeRed'][new_cat-1]
                    table = self.db.new_table(options= {'align':'l', 'color':f'{color}'}) 
                    table.field_names = [f'Features{category[new_cat-1]}', 'MAE', 'MSE', 'RMSE', 'R²']    
                    last = new_cat 
                    
            table.add_row(row)

        print()
        print(table)
        return sorted_data

    def show_plot(self):
        if self.field_scores:
            r_values = [(feat['features'], round(feat['scores']['R2'],3)) for feat in self.field_scores.values()]
            r_values = [feat for feat in r_values if len(feat[0]) == 1]
            r_values = sorted(r_values, key =lambda x: -x[1])[:self.plot_limit]

            # Data for the three elements
            labels = [label[0][0] for label in r_values]
            r2_values = [value[1] for value in r_values]

            # Create a figure with a soft grey background
            fig, ax = plt.subplots(figsize=(6, 6))
            fig.patch.set_facecolor('#909090')  # Set the background color of the figure

            # Create a pie chart
            ax.pie(r2_values, labels=labels, autopct='%1.1f%%', startangle=140)

            # Set the title
            plt.title('Proportions with Respect to R-squared (r²)')

            # Display the pie chart
            plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is drawn as a circle.
            plt.show()
 
class SimpleMiner_Linear(SimpleMinerBaseObject):

    def __init__(self, db_object, table_name, color=None) -> None:
        super().__init__(db_object, table_name, color)

        self.model_type = "Linear"
    
    def one_hot_encode_new_data(self, input_data):
        super().one_hot_encode_new_data(input_data)
        # Ensure the new data has the same columns as the one-hot encoded training data

        new_data = {}

        for col in self.one_hot_columns:
            if col not in input_data:
                # If the column is missing, add it with all zeros
                new_data[col] = False
            else:
                new_data[col] = input_data[col]
        
        return pd.DataFrame(new_data)
    
    def define_model_features(self, features, target=None):
        super().define_model_features(features, target)

        if target is None:
            if self.current_target:
                target = [self.current_target]
            else:
                raise ValueError("Missing target.")
        
        features = self.current_features

        # Define the features (independent variables) and the target variable (Salary)
        X = self.data_frame[features]
        y = self.data_frame[target]

        # Examine Unique Values and Apply One-Hot Encoding
        self.convert = {}
        for feature in features:
            unique_values = self.data_frame[feature].unique()

            if len(unique_values) == 2: #boolean mapping
                new_mapping = {'input_value': unique_values[0], 'new_value': True , 'new_column': f"{feature}_{unique_values[0]}"}
                self.convert[feature] = new_mapping
            
            # If the feature its just a label, then encode numeric / boolean
            if isinstance(self.data_frame[feature][0], str):
                # Perform one-hot encoding
                X = pd.get_dummies(X, columns=[feature], drop_first=True)
                #multiple selection mapping:
                self.convert[feature] = "multiple_selection"

        self.one_hot_columns = X.columns
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the linear regression model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Make predictions on the test set
        self.y_pred = self.model.predict(X_test)

        # Evaluate the model
        self.mae = mean_absolute_error(self.y_test, self.y_pred)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.rmse = np.sqrt(self.mse)
        self.r2 = r2_score(self.y_test, self.y_pred)
        
    def predict(self, input_values):
        super().predict(input_values)

        if self.model:

            mapped_data = {}

            #Add columns for boolean transformations (i.e Gender)
            for field in input_values:
                if field in self.convert:                               #for discreet values
                    if self.convert[field] == "multiple_selection":
                        try:
                            mapped_data[field + "_" + input_values[field]] = True
                        except Exception as e:
                            print(cs(e, color='yellow'))
                            return
                        
                    elif input_values[field] == self.convert[field]['input_value']: #for boolean clasification
                        mapped_data[self.convert[field]['new_column']] = [self.convert[field]['new_value']]
                    else:
                        mapped_data[self.convert[field]['new_column']] = [not self.convert[field]['new_value']]
                else:
                    mapped_data[field] = [input_values[field]]
            
            encoded_data = self.one_hot_encode_new_data(mapped_data)

            predicted_output = self.model.predict(encoded_data)

        return predicted_output[0][0]
    
class SimpleMiner_ANN(SimpleMinerBaseObject):

    def __init__(self, db_object, table_name, color=None):
        super().__init__(db_object, table_name, color)

        self.model_type = "ANN-seq"
        self.model_current_features = []

    def build_model(self, num_features):
        # Create and compile a dynamic model with a specified number of input features
        input_shape = (None, num_features) 
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape = input_shape ), 
            tf.keras.layers.Dense(64, activation='relu'),  # Additional hidden layers
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)  # Single output neuron for regression
            ])  
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def adjust_input_shape(self, num_features):
        # Method to adjust the input shape based on the number of features
        self.model.layers[0].batch_input_shape = (None, num_features)


    def train_model(self, X_train, y_train, epochs):
        # Train the model   
        self.model.fit(X_train, y_train, epochs=epochs, verbose=0) #verbose 0 deactivates log
        
    def predict(self, X_test):
        # Make predictions using the model
        return self.model.predict(X_test)    #verbose 0 deactivates log

    def define_model_features(self, features, target=None):
        super().define_model_features(features, target)

        print("Features:::", features)

        if target is None:
            if self.current_target:
                target = self.current_target
            else:
                raise ValueError("Missing target.")
        
        # Examine Unique Values and Apply One-Hot Encoding
        self.convert = {}
        one_hot_columns = []

        # Create an instance of the OneHotEncoder
        encoder = OneHotEncoder(sparse=False)

        print(self.data_frame)

        for feature in features:
            unique_values = self.data_frame[feature].unique()

            data_type = self.data_frame[feature].dtype
            # Binary field input - (i.e 'Gender')
            if len(unique_values) == 2:  # Boolean mapping
                new_mapping = {'input_value': unique_values[0], 'new_value': 1, 'new_column': "same"}
                self.convert[feature] = new_mapping

                le = LabelEncoder()
                self.data_frame[feature] = le.fit_transform(self.data_frame[feature])  # Male: 1, Female: 0

            # If the feature is just a label, then encode (i.e. 'Job Title')
            elif isinstance(self.data_frame[feature][0], str):
                # Extract the feature data and reshape for encoding
                feature_data = self.data_frame[[feature]]
                encoded_data = encoder.fit_transform(feature_data)

                # Get the one-hot column names
                encoded_feature_names = encoder.get_feature_names_out([feature])

                # Define new column names based on your logic
                new_column_names = [str(col).replace("'", '').replace(' ', '_') for col in encoded_feature_names]

                # Store the column names
                one_hot_columns.extend(new_column_names)

                # Create a DataFrame with the encoded data and new column names
                encoded_df = pd.DataFrame(encoded_data, columns=new_column_names)
                print(encoded_data)

                # Drop the original column and concatenate the encoded DataFrame
                self.data_frame = self.data_frame.drop(columns=[feature])
                self.data_frame = pd.concat([self.data_frame, encoded_df], axis=1)
                self.convert[feature] = "multiple_selection"

            # Standardize numeric values for ANN
            elif data_type in (int, float, np.float64, np.int64):
                scaler = StandardScaler()
                self.data_frame[feature] = scaler.fit_transform(self.data_frame[feature].values.reshape(-1, 1))
                # Create or adjust the model based on the number of features

        # Prepare the data
        X = self.data_frame.drop(columns=target)  # Input features (excluding the target)
        y = self.data_frame[target].astype(np.float32)  # Target variable

        self.one_hot_columns = X.columns
        total_number_of_features = len (self.one_hot_columns)

        # Specify random values
        tf.random.set_seed(42)
        np.random.seed(42)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # Build the model, and re build when features have changed
        features_have_changed = not (all(f.capitalize() in self.model_current_features for f in features) and \
                                len(self.current_features) == len(self.model_current_features))
        
        if features_have_changed:
            self.model = self.build_model( num_features = total_number_of_features )
            self.model_current_features = self.current_features

        # Train the model with adjusted input shape
        self.train_model(X_train, y_train, epochs=50)

        # Make predictions
        y_pred = self.predict(X_test)

        #print(y_pred) # Check the format of the the prediction

        # Reshape y_pred based on the shape of y_test
        if y_test.shape != y_pred.shape:
            y_pred = y_pred.reshape(y_test.shape)

        # Evaluate the model
        mae = tf.keras.metrics.mean_absolute_error(y_test, y_pred).numpy().mean()
        mse = tf.keras.metrics.mean_squared_error(y_test, y_pred).numpy().mean()
        rmse = np.sqrt(mse)

        # Calculate R-squared (R2)
        mean_y_test = np.mean(y_test)
        total_sum_of_squares = np.sum(np.square(y_test - mean_y_test))

        # Calculate the residual sum of squares
        residual_sum_of_squares = np.sum(np.square(y_test - y_pred))

        r2 = 1.0 - (residual_sum_of_squares / total_sum_of_squares)

        # Store the scores as single float values
        self.mae = float(mae)
        self.mse = float(mse)
        self.rmse = float(rmse)
        self.r2 =  float(r2.iloc[0])




    def show_plot(self):
        
        # Create a line plot for the relationship between actual and predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(self.actual_values, label='Actual', marker='o')
        plt.plot(self.predicted_values, label='Predicted', marker='x')
        plt.xlabel('Data Points')
        plt.ylabel('Values')
        plt.legend()
        plt.title('Actual vs. Predicted Values')
        plt.grid(True)
        plt.show()