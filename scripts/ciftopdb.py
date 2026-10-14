from Bio.PDB import MMCIFParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.StructureBuilder import StructureBuilder
import string
import copy


class ProteinOnlySelect(Select):
    def accept_residue(self, residue):

        # 只保留标准氨基酸
        if is_aa(residue, standard=True):
            return True

        return False


def rename_chain_ids(structure):
    """
    将长 chain id 映射为单字符
    """

    allowed_ids = list(string.ascii_uppercase +
                       string.ascii_lowercase +
                       string.digits)

    chain_map = {}

    current_idx = 0

    for model in structure:
        for chain in model:

            old_id = chain.id

            if old_id not in chain_map:

                if current_idx >= len(allowed_ids):
                    raise ValueError("Too many chains for PDB format")

                chain_map[old_id] = allowed_ids[current_idx]
                current_idx += 1

            chain.id = chain_map[old_id]

    print("Chain Mapping:")
    for k, v in chain_map.items():
        print(f"{k} -> {v}")

    return structure


def cif_to_clean_pdb(input_cif, output_pdb):

    parser = MMCIFParser(QUIET=True)

    structure = parser.get_structure("protein", input_cif)

    # 修复 chain id
    structure = rename_chain_ids(structure)

    io = PDBIO()
    io.set_structure(structure)

    io.save(output_pdb, ProteinOnlySelect())

    print(f"\nSaved cleaned pdb to: {output_pdb}")


if __name__ == "__main__":

    input_cif = "/data/wenhao/Protein-Hunter-server/AAV/pdb/4LRH.cif"
    output_pdb = "/data/wenhao/Protein-Hunter-server/AAV/pdb/4LRH.pdb"

    cif_to_clean_pdb(input_cif, output_pdb)