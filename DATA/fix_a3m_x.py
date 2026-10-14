from pathlib import Path

def process_a3m(input_path, output_path):
    print(f"Reading {input_path}...")
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    processed_lines = []
    is_query = False
    query_count = 0
    
    for line in lines:
        if line.startswith(">"):
            processed_lines.append(line)
            # 假设第一条序列是查询链
            query_count += 1
            is_query = (query_count == 1)
        else:
            seq = line.strip()
            if not is_query:
                # 除了查询链，将其余序列的 X 替换为 -
                seq = seq.replace('X', '-')
                seq = seq.replace('x', '-')
            processed_lines.append(seq + "\n")
            
    with open(output_path, 'w') as f:
        f.writelines(processed_lines)
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    input_file = "/data/wenhao/Protein-Hunter-server/MSA/9OZL_A_clean.a3m"
    output_file = "/data/wenhao/Protein-Hunter-server/MSA/9OZL_A_clean_fixed.a3m"
    process_a3m(input_file, output_file)
