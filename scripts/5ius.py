from Bio.PDB import PDBParser, PDBIO, Select

class CleanAndSplitSelect(Select):
    """
    自定义选择类：过滤掉溶剂/水，并选择特定链
    """
    def __init__(self, keep_chains):
        self.keep_chains = keep_chains

    def accept_chain(self, chain):
        # 只保留指定的链
        return chain.get_id() in self.keep_chains

    def accept_residue(self, residue):
        # 1. 剔除水分子 (Residue ID 通常为 'W' 或 'HOH')
        # 2. 剔除非蛋白质/核酸的异质原子 (HETATM)
        # get_resname() 可以精准过滤常见溶剂如 GLY, DMS 等，这里采用通用的非空判断
        res_id = residue.get_id()[0]
        
        # ' ' 代表标准氨基酸或核苷酸，'H_' 开头通常是 HETATM (溶剂、配体等)
        # 如果你想保留配体只删掉水，可以将条件改为: return res_id != 'W'
        return res_id == ' '

def process_pdb(input_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', input_file)
    io = PDBIO()
    io.set_structure(structure)

    # 提取 AC 链
    print(f"正在生成 AC 链文件...")
    io.save("chains_AC.pdb", CleanAndSplitSelect(['A', 'C']))

    # 提取 BD 链
    print(f"正在生成 BD 链文件...")
    io.save("chains_BD.pdb", CleanAndSplitSelect(['B', 'D']))

    print("处理完成！生成文件：chains_AC.pdb, chains_BD.pdb")

if __name__ == "__main__":
    # 在这里输入你的文件名
    pdb_filename = "/data/wenhao/Protein-Hunter-server/example_myself/5IUS.pdb" 
    try:
        process_pdb(pdb_filename)
    except FileNotFoundError:
        print(f"错误：找不到文件 {pdb_filename}，请检查路径。")