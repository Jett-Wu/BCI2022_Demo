import pandas as pd
import scipy.io as scio
import numpy as np
from sklearn import svm


def load_data(data_paths):
    """
    加载训练和测试数据
    Load training and test data
    """
    data = [scio.loadmat(path) for path in data_paths]
    return data[0]['train_data'], data[1]['train_label'], data[2]['test_data']


def extract_features(data):
    """
    提取统计特征：均值、最大值、标准差
    Extract statistical features: mean, max, std
    """
    features = np.concatenate([
        np.mean(data, axis=-1),
        np.max(data, axis=-1),
        np.std(data, axis=-1)
    ], axis=1).reshape(data.shape[0], -1)
    return features


def save_predictions(predictions, output_paths):
    """
    保存预测结果到CSV和MAT文件
    Save predictions to CSV and MAT files
    """
    # 保存CSV Save to CSV
    trial_id = list(range(1, len(predictions) + 1))
    pd.DataFrame({
        'TrialId': trial_id,
        'Label': predictions
    }).to_csv(output_paths['csv'], index=False)

    # 保存MAT Save to MAT
    scio.savemat(output_paths['mat'], {'test_label': predictions.reshape(-1, 1)})


def train_fNIRS():
    """
    训练fNIRS数据的SVM分类器并保存预测结果
    Train SVM classifier for fNIRS data and save predictions
    """
    # 定义文件路径 Define file paths
    data_paths = [
        './data/train_data.mat',
        './data/train_label.mat',
        './data/test_data.mat'
    ]
    output_paths = {
        'csv': 'submission.csv',
        'mat': 'submission.mat'
    }

    # 加载数据并提取特征 Load data and extract features
    train_data, train_label, test_data = load_data(data_paths)
    train_features = extract_features(train_data)
    test_features = extract_features(test_data)

    # 训练模型并预测 Train model and predict
    clf = svm.SVC(gamma='auto', C=1, kernel='rbf', probability=True, random_state=42)
    clf.fit(train_features, train_label.ravel())
    
    # 评估和预测 Evaluate and predict
    train_accuracy = clf.score(train_features, train_label.ravel())
    print(f'训练集准确率为 Training accuracy: {train_accuracy:.4f}')
    
    test_pred = clf.predict(test_features)
    save_predictions(test_pred, output_paths)
    print('预测结果已保存至 Predictions saved to: submission.csv 和 submission.mat')


if __name__ == '__main__':
    train_fNIRS()
