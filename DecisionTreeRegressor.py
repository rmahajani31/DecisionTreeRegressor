import numpy as np
import pandas as pd
import math
from Tree import *

class DecisionTreeRegressor:
    def __init__(self, filepath, split_criteria, max_depth):
        self.filepath = filepath
        self.split_criteria = split_criteria
        self.max_depth = max_depth
        self.data = self.load_data()
        self.tree = Tree()

    def load_data(self):
        return pd.read_csv(self.filepath)

    def calculate_standard_deviation(self, data):
        return data["class"].std

    def get_all_splits(self, data, col):
        col_vals = data[col].unique()
        return self.get_all_splits_helper(col_vals, 0, np.array([]), np.array([]))

    def get_all_splits_helper(self, col_vals, col_ind, left_split, right_split):
        if col_ind == len(col_vals):
            return np.array([[left_split, right_split]])
        left_result = np.array([[[]]])
        right_result = np.array([[[]]])
        if left_split.shape[0] < len(col_vals) - 1:
            left_result = self.get_all_splits_helper(col_vals, col_ind+1, np.append(left_split, col_vals[col_ind]), right_split)
        if right_split.shape[0] < len(col_vals) - 1:
            right_result = self.get_all_splits_helper(col_vals, col_ind+1, left_split, np.append(right_split, col_vals[col_ind]))
        if left_result.size > 0 and right_result.size > 0:
            return np.concatenate((left_result, right_result), axis=0)
        return left_result if left_result.size > 0 else right_result


    def get_average(self, data):
        return data["class"].mean()

    def calculate_std_gain(self, original_std, data, col):
        total_count = data.shape[0]
        splits = self.get_all_splits(data, col)
        max_gain = 0
        max_left_split = np.array([])
        max_right_split = np.array([])
        for split in splits:
            split_data_left = data.loc[data[col].isin(split[0]), :]
            split_data_right = data.loc[data[col].isin(split[1]), :]
            std = split_data_left.shape[0] / total_count * self.calculate_standard_deviation(split_data_left) + split_data_right.shape[0] / total_count * self.calculate_standard_deviation(split_data_right)
            gain = original_std - std
            if gain > max_gain:
                max_gain = gain
                max_left_split = split_data_left
                max_right_split = split_data_right
        return (max_gain, max_left_split, max_right_split)

    def get_max_col(self, data):
        cols = list(data)
        data_std = self.calculate_standard_deviation(data)
        max_criteria_val = 0
        max_col = 0
        max_left_split = np.array([])
        max_right_split = np.array([])
        for index, col in enumerate(cols):
            if index < len(cols) - 1:
                (criteria_val, left_split, right_split) = self.calculate_std_gain(data_std, data, col)
                if criteria_val > max_criteria_val:
                    max_criteria_val = criteria_val
                    max_left_split = left_split
                    max_right_split = right_split
                    max_col = index
        return (max_col, max_left_split, max_right_split)

    def train(self):
        self.tree = Tree(self.train_with_depth(self.data, self.max_depth))

    def train_with_depth(self, data, maxDepth):
        if maxDepth == 0:
            leaf = Node()
            leaf.value = self.get_average(data)
            return leaf
        else:
            cur = Node()
            (max_col, max_left_split, max_right_split) = self.get_max_col(data)
            cols = list(data)
            cur.col = max_col
            cur.value = self.get_average(data)
            if max_left_split.shape[0] > 0:
                cur.add_child(self.train_with_depth(max_left_split, maxDepth-1))
            if max_right_split.shape[0] > 0:
                cur.add_child(self.train_with_depth(max_right_split, maxDepth-1))
            return cur

    def predict(self, sample):
        return self.predict_with_node(sample, self.tree.head)

    def predict_with_node(self, sample, cur):
        if cur.col == -1:
            return cur.value
        if sample[cur.col] == 0 and len(cur.children) > 0:
            return self.predict_with_node(sample, cur.children[0])
        if sample[cur.col] == 1 and len(cur.children) > 1:
            return self.predict_with_node(sample, cur.children[1])
        return cur.value