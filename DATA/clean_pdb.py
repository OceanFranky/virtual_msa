from Bio.PDB import PDBParser, PDBIO, Select

def clean_pdb(input_pdb, output_pdb):
    # 定义一个内部过滤器，只放行标准氨基酸
    class ProteinOnly(Select):
        def accept_residue(self, residue):
            if residue.id[0] == " ":
                return 1  # 1 代表保留
            else:
                return 0  # 0 代表丢弃 (去掉水分子、离子等 HETATM)

    # 读取原始结构
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("MyProtein", input_pdb)
    
    # 保存纯净版结构
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, ProteinOnly())
    print(f"清洗完成！已生成纯净多肽链文件：{output_pdb}")

# 调用示例：
# clean_pdb("1TAQ.pdb", "1TAQ_clean.pdb")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="清理PDB文件，去除水分子、离子和配体，仅保留标准氨基酸。")
    parser.add_argument("input_pdb", help="输入的PDB文件路径")
    parser.add_argument("output_pdb", help="输出的PDB文件路径")
    
    args = parser.parse_args()
    clean_pdb(args.input_pdb, args.output_pdb)