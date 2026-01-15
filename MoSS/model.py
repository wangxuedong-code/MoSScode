import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import pandas as pd
from sklearn.model_selection import KFold
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import os
import random
import math
import xgboost as xgb
from statistics import mode
from sklearn.kernel_approximation import AdditiveChi2Sampler
pd.set_option('display.max_columns', None)  


class TreeNode:
    def __init__(self, feature=None, classifier=None, anova=None, weight=None):
        self.label = None
        # self.probability = None
        self.children = []
        self.feature = feature
        self.classifier = classifier
        self.anova = anova
        self.weight = weight
        self.probability = None


def information_entropy(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def most_frequent_label(resul):
    mode_result = resul.mode().iat[0]
    return mode_result


def compute_weight_by_accuracy(classifier, df, feature, anova):
    if 'label' in feature:
        feature = feature
    else:
        feature.append('label')
    df = df.dropna(subset=feature)
    df = df[feature]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    if X.shape[0] == 0 :
        accuracy = 0.5
    else:
        X = anova.transform(X)
        predictions = classifier.predict(X)
        accuracy = np.mean(predictions == y)
    return accuracy


def calculate_distribution(column_data):
    unique_values = sorted(column_data.unique())
    if len(unique_values) > 2 or not set(unique_values).issubset({0, 1}):
        raise ValueError("列数据必须仅包含 0 和 1")
    total_count = len(column_data)
    count_0 = (column_data == 0).sum()
    count_1 = (column_data == 1).sum()
    prob_0 = round(count_0 / total_count, 3) if total_count > 0 else 0
    prob_1 = round(count_1 / total_count, 3) if total_count > 0 else 0
    return [prob_0, prob_1]



def generate_tree(df, df3, feature, δ, n, i, q_new, para_s, para_b):
    node = TreeNode()
    q = df.shape
    abnormal = len(q_new)
    node.probability = calculate_distribution(df['label'])
    entropy = information_entropy(df['label'])
    if entropy < δ or len(df) < n or len(q_new) == 0:
        node.label = most_frequent_label(df['label'])
        abnormal = 1
        return node, abnormal

    columns_to_drop = [col for col in feature if col != 'label']
    df = df.drop(columns=columns_to_drop)
    df_feature = df.columns.tolist()
    subset_data, certain_f, q_new, syb_data = view_selection(df, feature, df_feature, q_new)

    if 'label' in certain_f:
        certain_f.remove('label')
        node.feature = certain_f.copy()
    else:
        node.feature = certain_f.copy()

    labels = subset_data['label']
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        node.probability = calculate_distribution(labels)
        node.label = most_frequent_label(labels)
        return node, abnormal

    if syb_data == 0:
        node.label = most_frequent_label(df['label'])
        return node, abnormal

    classifier, anova = train_classifier1(subset_data, para_s, para_b, q)

    node.classifier = classifier
    node.anova = anova
    node.weight = compute_weight_by_accuracy(classifier, df3, certain_f, anova)

    i = i + 1

    for y in set(df['label']) | {'unknown'}:
        if 'label' not in certain_f:
            certain_f.append('label')
        dfy = samples_classified_to_y(df, classifier, y, certain_f, anova)
        m = dfy.shape
        if not dfy.empty:
            if y != 'unknown':
                child_node, abnormal = generate_tree(dfy, df3, certain_f, δ, n, i, q_new, para_s, para_b)
                node.children.append(child_node)
            else:
                child_node, abnormal = generate_tree(dfy, df3, certain_f, δ, n, i, q_new, para_s, para_b)
                node.children.append(child_node)
        if dfy.empty:
            child_node = generate_tree2(dfy, df3, y)
            node.children.append(child_node)
    return node, abnormal


def generate_tree2(df, df3, y):
    node = TreeNode()
    node.label = 0
    return node

def remove_high_missing_columns(df, threshold):
    missing_ratios = df.isnull().mean()
    columns_to_keep = missing_ratios[missing_ratios <= threshold].index
    new_data = df[columns_to_keep]
    return new_data

def select_features(df, lists, k):
    missing_ratios = df.isnull().mean()
    columns_to_keep = missing_ratios[missing_ratios <= 0.5].index
    columns_to_keep = columns_to_keep.tolist()
    selected = []
    if not columns_to_keep:
        return selected
    candidate_list = lists.copy()

    max_loopss = 50
    loop_counts = 0

    while len(selected) < k and candidate_list and loop_counts < max_loopss:
        loop_counts += 1
        weights = [entry['Probability'] for entry in candidate_list]
        selected_entry = random.choices(candidate_list, weights=weights, k=1)[0]
        feature_name = selected_entry['F']
        if feature_name in columns_to_keep:
            selected.append(selected_entry)
            candidate_list = [entry for entry in candidate_list if entry['F'] != feature_name]
    return selected


def view_selection(df, feature, df_feature, q_new):
    list1 = []
    label = ["label"]
    if 'label' in feature:
        feature.remove('label')

    for Fi in set(df_feature) - set(feature) - set(label):
        sui = symmetrical_uncertainty(df[Fi], df['label'])
        list1.append({'F': Fi, 'SU': sui})

    list1.sort(key=lambda x: x['SU'], reverse=True)
    total_su = sum(entry['SU'] for entry in list1)

    if total_su == 0:
        prob = 1.0 / len(list1) if list1 else 0
        for entry in list1:
            entry['Probability'] = prob
    else:
        for entry in list1:
            entry['Probability'] = entry['SU'] / (total_su / 4)

    for item in list1:
        item['SU'] = round(item['SU'], 4)
        item['Probability'] = round(item['Probability'], 4)

    prob_list = [entry['Probability'] for entry in list1]
    total_prob = sum(prob_list)

    if total_prob <= 0 and list1:
        uniform_prob = 1.0 / len(list1)
        for entry in list1:
            entry['Probability'] = round(uniform_prob, 4)

    certain_f = set()
    subset = pd.DataFrame()

    fetched = False
    features_select = None
    max_loops = 5
    loop_count = 0

    while q_new and loop_count < max_loops:
        loop_count += 1
        try:
            if not fetched:
                features_select = q_new.pop(0)
                fetched = True
            features = select_features(df, list1, features_select)

            for d in features:
                certain_f.add(d['F'])
            certain_f = list(certain_f) + label
            subset = df[certain_f].copy()
            subset = subset.dropna()
            if not subset.empty:
                break
            features_select -= 1
            if features_select < 2:
                break
        except Exception as e:
            if fetched and features_select >= 2:
                features_select -= 1

    subset_features = subset.drop(columns=['label'], errors='ignore')
    if subset_features.empty or subset_features.shape[1] == 0:
        syb_data = 0
    else:
        syb_data = 1


    return subset, certain_f, q_new, syb_data


def train_classifier1(df, par_s, par_b, q):
    # print(f'df{df}')
    dff = df.drop('label', axis=1)
    labels = df['label']
    if dff.shape[1] == 0:
        print(df)
        raise ValueError("No features provided for training.")
    classifier = xgb.XGBClassifier(
        objective='multi:softmax',
        eta=0.01,
        max_depth=7,
        num_class=2,
        n_estimators=300)

    label_counts = Counter(labels)
    majority_class = max(label_counts, key=label_counts.get)
    minority_class = min(label_counts, key=label_counts.get)

    if label_counts[majority_class] >= 1.5 * label_counts[minority_class] and \
            label_counts[0] >= 5 and label_counts[1] >= 5:
        dff, labels = data_smote(dff, labels, par_s)

    anova = AdditiveChi2Sampler(sample_steps=par_b)
    anova_train = anova.fit_transform(dff, labels)
    classifier.fit(anova_train, labels)
    return classifier, anova


def data_smote(dfs, dfl, par_s):
    label_counts = dfl.value_counts()
    count_negative_samples = label_counts.min()
    count_positive_samples = label_counts.max()
    majority_class = label_counts.idxmax()
    minority_class = label_counts.idxmin()

    max_positive_samples = int(count_positive_samples / 1.5)
    sampling_strategy = {majority_class: count_positive_samples, minority_class: max_positive_samples}
    smote = SMOTE(random_state=42, k_neighbors=par_s, sampling_strategy=sampling_strategy)
    dfss, dfll = smote.fit_resample(dfs, dfl)

    return dfss, dfll


def samples_classified_to_y(df, classifier, y, certain_f, anova):
    if y == 0 or y == 1:
        df = df.dropna(subset=certain_f)
        m1 = df.shape
        # print(f"视图标签: {y}")
        dff = df[certain_f].drop('label', axis=1)
        dff = anova.transform(dff)
        predictions = classifier.predict(dff)
        dy = df[predictions == y].copy()
        return dy
    else:
        dy = df[df[certain_f].isnull().any(axis=1)]
        return dy


def symmetrical_uncertainty(x, y):
    mutual_information = mutual_information_between(x, y)
    entropy_x = information_entropy(x)
    entropy_y = information_entropy(y)
    su = 2 * mutual_information / (entropy_x + entropy_y)
    return su


def mutual_information_between(x, y):
    joint_probability = pd.crosstab(x, y, margins=False, normalize=True)

    p_x = joint_probability.sum(axis=1)
    p_y = joint_probability.sum(axis=0)
    p_xy = joint_probability
    p_x[p_x == 0] = 1e-10
    p_y[p_y == 0] = 1e-10
    p_xy[p_xy == 0] = 1e-10
    mutual_information = np.sum(np.sum(p_xy * np.log2(p_xy / (np.outer(p_x, p_y)))))
    return mutual_information


def check_dataset(dataset):
    num_columns_without_label = len(dataset.columns) - 1  

    if num_columns_without_label <= 2 and dataset.drop(columns='label').isnull().any(axis=1).all():
        return 1
    return 0


def generate_view(df, features):
    subset = df[features].copy()
    return subset


def fill_missing_with_zero_for_binary_columns(data_df, specified_values):
    binary_columns = [col for col in data_df.columns if set(data_df[col].dropna().unique()) == set(specified_values)]
    data_df[binary_columns] = data_df[binary_columns].fillna(0)
    return data_df

def fusion(a, b):
    m1 = np.array(a).squeeze()
    m2 = np.array(b).squeeze()
    if m1.ndim != 1 or m2.ndim != 1 or len(m1) != len(m2):
        raise ValueError(f"输入数组维度错误！a维度：{m1.ndim}, b维度：{m2.ndim}，长度：{len(m1)}/{len(m2)}")
    k = 0
    for i in range(len(m1)):
        for j in range(len(m1)):
            k += m1[i] * m2[j]
    res = 0
    for q in range(len(m1)):
        res += m1[q] * m2[q]
    k = k - res
    denominator = 1 - k
    if abs(denominator) < 1e-8:
        denominator = 1e-8
    list_A = []
    for s in range(len(m1)):
        A = (m1[s] * m2[s]) / denominator
        list_A.append(A)
    list_A = np.nan_to_num(list_A, nan=0.0, posinf=1.0, neginf=0.0) 
    sum_A = np.sum(list_A)
    if sum_A < 1e-8:
        list2 = np.ones_like(list_A) / len(list_A)
    else:
        list2 = list_A / sum_A
    result = np.array(list2).squeeze()
    return result


def fusion_multi_dim(z):
    standardized = []
    for item in z:
        if item == []:
            continue
        arr = np.array(item).squeeze()
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 1:
            standardized.append(arr)
    if len(standardized) == 0:
        return np.array([0.5, 0.5])
    result = np.array(standardized)
    for i in range(result.shape[0] - 1):
        result[i + 1] = fusion(result[i], result[i + 1])
    result_last = result[-1].squeeze()
    return result_last


def print_tree(node, indent=""):
    print(indent + f"Feature: {node.feature}, Label: {node.label}, classifier: {node.classifier or 'None'}, 概率: {node.probability or 'None'}")

    for child in node.children:
        print_tree(child, indent + "  ")


def predict_tree(node, sample, pr_list_label):
    if not node.children:
        return node.label, pr_list_label

    current_feature = node.feature
    if 'label' in current_feature:
        current_feature.remove('label')
    current_classifier = node.classifier
    current_anova = node.anova
    current_data = sample[current_feature]

    predictions = 3

    if node.classifier is None:
        predictions = 'unknown'

    if current_data.isnull().any().any() or predictions == 'unknown':
        predictions = 'unknown'
        probabili = node.probability
        # print(node.probability)
        pr_list_label.append([np.array(probabili, dtype=np.float32)])
    else:
        current_data = current_anova.transform(current_data)
        predictions = current_classifier.predict(current_data)
        y_pred_proba = current_classifier.predict_proba(current_data)
        y_pred_proba_rounded = np.round(y_pred_proba, decimals=3)
        pr_list_label.append(y_pred_proba_rounded)

    if predictions == 0:
        next_node = node.children[0]
    elif predictions == 1:
        next_node = node.children[1]
    elif predictions == 'unknown':
        next_node = node.children[2]
    else:
        return node.label, pr_list_label

    return predict_tree(next_node, sample, pr_list_label)



def label(p_list, pr_list, r_list, s_predict):
    for root in r_list:
        pr_list_label = []
        predicted, predicteds = predict_tree(root, s_predict, pr_list_label)
        p_list.append(predicted)
        need_list = predicteds[-1:]
        pr_list.append(need_list)
    p_label = mode(p_list)
    return p_label, pr_list


def combine_weights(weights1, weights2):
    if len(weights1) != len(weights2):
        raise ValueError(f"两个权重列表的长度不一致: weights1({len(weights1)}), weights2({len(weights2)})")
    weights1 = np.array(weights1, dtype=float)
    weights2 = np.array(weights2, dtype=float)
    norm_weights1 = weights1 / weights1.sum() if weights1.sum() != 0 else np.zeros_like(weights1)
    norm_weights2 = weights2 / weights2.sum() if weights2.sum() != 0 else np.zeros_like(weights2)
    weights = norm_weights1 + norm_weights2
    return weights.tolist()


class GatedNetwork(nn.Module):
    def __init__(self, sub_models, feature_sets, top_k):
        super(GatedNetwork, self).__init__()
        self.sub_models = sub_models
        self.feature_sets = feature_sets
        self.top_k = top_k

    def forward(self, x):
        available_features = set(x.columns)
        weights1 = [len(available_features & feature_set) for feature_set in self.feature_sets]
        weights2 = []
        for model in self.sub_models:
            total_weight = 0
            for node in model.children:
                if node.weight is not None:
                    total_weight += node.weight
            weights2.append(total_weight)

        weights = combine_weights(weights1, weights2)
        top_k_indices = sorted(
            range(len(weights)), key=lambda i: weights[i], reverse=True
        )[:self.top_k]
        outputs_list = []
        for idx in top_k_indices:
            outputs_list.append(self.sub_models[idx])
        return outputs_list


def calculate_metrics(confusion_matrix):
    TP = confusion_matrix[1][1]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TN = confusion_matrix[0][0]

    Pre = TP / (TP + FP) if (TP + FP) != 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
    ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0

    return ACC, Pre, Recall, TNR


def models(df, para_c, para_k, para_a, para_n, para_s, para_b, breaklB, breaklS):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = os.path.normpath(os.path.join(
        current_dir,
        'data',
        df
    ))
    data = pd.read_excel(data)
    data_feature = data.columns.tolist()
    i = 1
    data = fill_missing_with_zero_for_binary_columns(data, specified_values=[1])
    data = fill_missing_with_zero_for_binary_columns(data, specified_values=[0, 1])
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    data = pd.DataFrame(scaled_data, columns=data.columns)

    kf = KFold(n_splits=10, shuffle=True, random_state=100)
    fold_indices = []
    for _, test_idx in kf.split(data):
        fold_indices.append(test_idx)

    accuracies = []
    all_results = {}
    precisions = []
    recalls = []
    auprcs = []
    Y_true = []
    Y_scores = []
    ii = 0
    Accs = []

    for start_idx in range(10):
        test_folds = [(start_idx + i) % 10 for i in range(breaklS)]
        remaining_folds = [i for i in range(10) if i not in test_folds]
        train_folds = remaining_folds[:breaklB]
        train_indices = np.concatenate([fold_indices[i] for i in train_folds])
        test_indices = np.concatenate([fold_indices[i] for i in test_folds])
        train_unique = np.unique(train_indices)
        test_unique = np.unique(test_indices)
        data1 = data.iloc[train_unique]
        data2 = data.iloc[test_unique]

        data1, data3 = train_test_split(data1, test_size=0.2, random_state=108)
        feature_sets = []

        root_list = []
        for _ in range(para_c):
            current = data1.shape[1] - 1
            # print(current)
            q = []
            max_loops = 5
            loop_count = 0
            while loop_count < max_loops:
                loop_count += 1
                current = current / 2
                floor_current = math.floor(current)
                if floor_current > 2:
                    q.append(floor_current)
                    current = floor_current
                elif floor_current == 2:
                    q.append(floor_current)
                    q.append(floor_current)
                    break
                else:
                    break

            root, abnormal = generate_tree(data1, data3, [], para_a, para_n, i, q, para_s, para_b)

            if abnormal != 0:
                root_list.append(root)

        list_length = len(root_list)

        for tree in root_list:
            tree_features = set()
            if tree is None:
                return
            if hasattr(tree, 'feature'):
                tree_features.update(tree.feature)
            feature_sets.append(tree_features)

        top_k = int(para_k * para_c)
        if top_k > list_length:
            top_k = list_length

        if top_k == 0:
            return "None", "None", "None", "None", np.float64(100.0)
        gated_network = GatedNetwork(root_list, feature_sets, top_k)

        conf_matrix2 = [[0, 0],
                        [0, 0]]

        conf_matrix3 = [[0, 0],
                        [0, 0]]

        y_true = []
        y_scores = []

        for index, row in data2.iterrows():
            sample_to_predict = pd.DataFrame(row).T
            target = int(sample_to_predict['label'].values[0])  # 将目标转换为整数
            sample_to_predict.drop(columns=['label'], inplace=True)
            predicted_list = []
            probability = []
            root_list = gated_network(sample_to_predict)

            predicted_label1, probability_list = label(predicted_list, probability, root_list, sample_to_predict)
            probability_list = [item for item in probability_list if item != []]
            ds2 = fusion_multi_dim(probability_list)
            predicted_label2 = np.argmax(ds2)
            ds2_flat = np.array(ds2).flatten()
            y_true.append(target)
            y_scores.append(ds2_flat[1])

            conf_matrix2[int(target)][int(predicted_label2)] += 1

        for index, row in data3.iterrows():
            sample_to_predict3 = pd.DataFrame(row).T
            target = int(sample_to_predict3['label'].values[0])  # 将目标转换为整数
            sample_to_predict3.drop(columns=['label'], inplace=True)
            predicted_list3 = []
            probability3 = []
            root_list = gated_network(sample_to_predict3)
            predicted_label1, probability_list = label(predicted_list3, probability3, root_list, sample_to_predict3)
            probability_list = [item for item in probability_list if item != []]
            ds2 = fusion_multi_dim(probability_list)
            predicted_label2 = np.argmax(ds2)
            conf_matrix3[int(target)][int(predicted_label2)] += 1

        auprc = average_precision_score(y_true, y_scores)
        acc, pre, rec, tnr = calculate_metrics(conf_matrix2)
        # print(conf_matrix2)
        Acc, _, _, _ = calculate_metrics(conf_matrix3)

        fold_results = {
            "Accuracy": acc,
            "Precision": pre,
            "Recall": rec,
            "AUPRC": auprc}

        all_results[f"Fold_{start_idx}"] = fold_results

        y_true = [f"{num:.3f}" for num in y_true]
        y_scores = [f"{num:.3f}" for num in y_scores]
        Y_true.append(y_true)
        Y_scores.append(y_scores)
        Accs.append(Acc)
        accuracies.append(acc)
        precisions.append(pre)
        recalls.append(rec)
        auprcs.append(auprc)

    Y_sample = {
        "Y_true": Y_true,
        "Y_scores": Y_scores
    }

    metrics_mean = {
        "Accuracy": np.mean(accuracies),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "AUPRC": np.mean(auprcs)
    }

    Accuracy = np.mean(Accs)

    Accuracy = Accuracy.round(3)

    metrics_std = {
        "Accuracy": np.std(accuracies),
        "Precision": np.std(precisions),
        "Recall": np.std(recalls),
        "AUPRC": np.std(auprcs)
    }

    return all_results, metrics_mean, metrics_std, Y_sample, Accuracy













