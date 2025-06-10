import csv
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# 自动选择设备和精度
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

# 加载 BERT 模型，用于计算句子嵌入
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)


# 读取数据库的真实类别（真实标签）CSV文件
def read_ground_truth(ground_truth_file):
    ground_truth = {}
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            category = row['category']
            ground_truth[filename] = category
    return ground_truth


# 读取处理后的结果CSV文件
def read_processed_results(processed_file):
    processed_results = {}
    with open(processed_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            filename = row[0]
            description = row[1]
            processed_results[filename] = description
    return processed_results


# 计算句子相似度
def calculate_similarity(description, category):
    # 将描述和类别文本转换为嵌入向量
    embeddings = model.encode([description, category])

    # 计算余弦相似度
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity


def calculate_similarity_with_model(description, category, model):
    embeddings = model.encode([description, category])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity


# 多线程，每个线程有自己的模型
def compare_results(ground_truth, processed_results, similarity_threshold=0.4, num_workers=2):
    all_categories = set(ground_truth.values())
    items = [(filename, description) for filename, description in processed_results.items() if filename in ground_truth]

    def worker(sub_items):
        # 每个线程独立加载模型
        local_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        correct = 0
        detailed = []
        for filename, description in sub_items:
            true_category = ground_truth[filename]
            similarities = {category: calculate_similarity_with_model(description, category, local_model) for category in all_categories}
            predicted_category = max(similarities, key=similarities.get)
            max_similarity = similarities[predicted_category]
            is_correct = (predicted_category == true_category)
            detailed.append({
                'Filename': filename,
                'Generated Description': description,
                'True Category': true_category,
                'Predicted Category': predicted_category,
                'Max Similarity': max_similarity,
                'Match': 'Yes' if is_correct else 'No'
            })
            if is_correct:
                correct += 1
        return correct, len(sub_items), detailed

    # 均分任务
    chunk_size = (len(items) + num_workers - 1) // num_workers
    chunks = [items[i*chunk_size:(i+1)*chunk_size] for i in range(num_workers)]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(worker, chunks))

    correct_count = sum(r[0] for r in results)
    total_count = sum(r[1] for r in results)
    all_details = []
    for r in results:
        all_details.extend(r[2])
    accuracy = correct_count / total_count if total_count > 0 else 0

    # 保存详细结果到CSV
    report_file = './classification_report.csv'
    with open(report_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Filename', 'Generated Description', 'True Category', 'Predicted Category', 'Max Similarity', 'Match'
        ])
        writer.writeheader()
        for row in all_details:
            writer.writerow(row)
    print(f"详细对比结果已保存到: {report_file}")

    return accuracy, correct_count, total_count


if __name__ == "__main__":
    # 设置路径
    ground_truth_file = './ESC-50-master/meta/esc50.csv'  # 修改为你的数据库文件路径
    processed_file = './audio_results.csv'  # 修改为处理后结果的CSV文件路径
    num_workers = 2  # 每个线程一个模型

    # 读取数据
    ground_truth = read_ground_truth(ground_truth_file)
    # 输出类别统计
    categories = set(ground_truth.values())
    print(f"共 {len(categories)} 个类别：")
    print(sorted(categories))
    processed_results = read_processed_results(processed_file)

    # 计算准确度
    accuracy, correct_count, total_count = compare_results(ground_truth, processed_results, num_workers=num_workers)

    # 输出总结
    print(f"总共有 {total_count} 个样本，正确分类的样本数量为 {correct_count}。")
    print(f"准确度：{accuracy * 100:.2f}%")
