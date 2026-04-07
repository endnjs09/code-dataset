import os
import json

def generate_jsonl_dataset(raw_dir, output_file):
    """
    raw_dir: .cpp 와 .json 쌍이 들어있는 폴더 경로
    output_file: 변환된 데이터를 저장할 .jsonl 파일 이름
    """
    if not os.path.exists(raw_dir):
        print(f"'{raw_dir}' 폴더를 찾을 수 없습니다. 폴더를 생성해주세요.")
        return

    # 하위 폴더까지 포함하여 모든 파일 목록 가져오기
    cpp_paths = []
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.endswith('.cpp'):
                cpp_paths.append(os.path.join(root, f))

    dataset_records = []

    print(f"🔍 {len(cpp_paths)}개의 C++ 코드를 찾았습니다. 짝꿍 JSON을 검색합니다...")

    for cpp_path in cpp_paths:
        cpp_file = os.path.basename(cpp_path)
        base_name = os.path.splitext(cpp_file)[0]

        if base_name.startswith("variant_"):
            json_file = base_name.replace("variant_", "trace_") + ".json"
        else:
            json_file = base_name + ".json"

        dir_name = os.path.dirname(cpp_path)

        if os.path.basename(dir_name) == 'code':
            parent_dir = os.path.dirname(dir_name) # '.../BFS_data'
            json_path = os.path.join(parent_dir, 'json', json_file)
        else:
            json_path = os.path.join(dir_name, json_file)

        # 짝꿍 json 파일이 존재한다면
        if os.path.exists(json_path):
            with open(cpp_path, 'r', encoding='utf-8') as cf:
                code_text = cf.read()

            with open(json_path, 'r', encoding='utf-8') as jf:
                # JSON 파일 텍스트를 그대로 가져오기
                trace_text = jf.read()

            # 모델에게 전달할 프롬프트 구조 만들기
            record = {
                "instruction": "아래 C++ 알고리즘 코드를 시각화하기 위해 완벽한 TraceLogger JSON 양식으로 번역하세요.",
                "input": code_text,
                "output": trace_text
            }
            dataset_records.append(record)
        else:
            print(f"경고: {cpp_file}의 짝꿍인 {json_file}을 찾지 못해 건너뜁니다.")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for record in dataset_records:
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"변환 완료! 총 {len(dataset_records)}개의 학습 데이터가 '{output_file}'에 포장되었습니다.")

if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))
    raw_folder = os.path.join(current_folder, "raw_data")
    output_jsonl = os.path.join(current_folder, "dataset.jsonl")

    generate_jsonl_dataset(raw_folder, output_jsonl)
