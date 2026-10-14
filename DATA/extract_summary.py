import os
import glob

# 你的数据目录
fasta_dir = "/data/wenhao/Protein-Hunter-server/DATA/small_protein_targets/fasta"
output_file = "/data/wenhao/Protein-Hunter-server/DATA/fasta_summary.txt"

fasta_files = glob.glob(os.path.join(fasta_dir, "*.fasta"))

if not fasta_files:
    print(f"在 {fasta_dir} 中未找到任何 .fasta 文件")
else:
    with open(output_file, "w") as out_f:
        for fasta_file in sorted(fasta_files):
            with open(fasta_file, "r") as f:
                lines = f.readlines()

            # 提取序列 ID（去掉开头的 >）
            seq_id = lines[0].strip().lstrip(">")

            # 提取序列（去掉换行符并连接）
            sequence = "".join(line.strip() for line in lines[1:])

            # 写入输出文件
            out_f.write(f"{seq_id}\t{sequence}\n")

    print(f"已提取 {len(fasta_files)} 条序列到 {output_file}")
