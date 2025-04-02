import csv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 加载 BERT 模型，用于计算句子嵌入
model = SentenceTransformer('all-MiniLM-L6-v2')


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


# 计算准确度并输出逐项对比结果
def compare_results(ground_truth, processed_results, similarity_threshold=0.4):
    comparison_results = []
    correct_count = 0
    total_count = 0

    for filename, description in processed_results.items():
        if filename in ground_truth:
            category = ground_truth[filename]

            # 计算相似度
            similarity = calculate_similarity(description, category)

            # 判断是否匹配
            is_correct = similarity >= similarity_threshold
            comparison_results.append({
                'Filename': filename,
                'Generated Description': description,
                'True Category': category,
                'Similarity': similarity,
                'Match': 'Yes' if is_correct else 'No'
            })

            if is_correct:
                correct_count += 1
            total_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0
    return comparison_results, accuracy, correct_count, total_count


if __name__ == "__main__":
    # 设置路径
    ground_truth_file = '/content/drive/MyDrive/ESC-50-master/meta/esc50.csv'  # 修改为你的数据库文件路径
    processed_file = '/content/drive/MyDrive/audio_results.csv'  # 修改为处理后结果的CSV文件路径

    # 读取数据
    ground_truth = read_ground_truth(ground_truth_file)
    processed_results = read_processed_results(processed_file)

    # 计算准确度并获取逐项对比结果
    comparison_results, accuracy, correct_count, total_count = compare_results(ground_truth, processed_results)

    # 输出逐项对比结果
    print("逐项对比结果：")
    for result in comparison_results:
        print(f"Filename: {result['Filename']}")
        print(f"Generated Description: {result['Generated Description']}")
        print(f"True Category: {result['True Category']}")
        print(f"Similarity: {result['Similarity']:.4f}")
        print(f"Match: {result['Match']}")
        print("-" * 50)

    # 输出总结
    print(f"总共有 {total_count} 个样本，正确分类的样本数量为 {correct_count}。")
    print(f"准确度：{accuracy * 100:.2f}%")
