#!/bin/bash

#make new directory for model parameters
#e.g.   bash get_model_params.sh "./model_params"

mkdir -p $1

#Original ProteinMPNN weights
aria2c -x 16 -s 16 -d $1 -o "proteinmpnn_v_48_002.pt" https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_002.pt
aria2c -x 16 -s 16 -d $1 -o "proteinmpnn_v_48_010.pt" https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_010.pt
aria2c -x 16 -s 16 -d $1 -o "proteinmpnn_v_48_020.pt" https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt
aria2c -x 16 -s 16 -d $1 -o "proteinmpnn_v_48_030.pt" https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_030.pt

#ProteinMPNN with num_edges=32
# aria2c -x 16 -s 16 -d $1 -o "proteinmpnn_v_32_002.pt" https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_32_002.pt
# aria2c -x 16 -s 16 -d $1 -o "proteinmpnn_v_32_010.pt" https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_32_010.pt
# aria2c -x 16 -s 16 -d $1 -o "proteinmpnn_v_32_020.pt" https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_32_020.pt
# aria2c -x 16 -s 16 -d $1 -o "proteinmpnn_v_32_030.pt" https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_32_030.pt

#LigandMPNN with num_edges=32; atom_context_num=25
aria2c -x 16 -s 16 -d $1 -o "ligandmpnn_v_32_005_25.pt" https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_005_25.pt
aria2c -x 16 -s 16 -d $1 -o "ligandmpnn_v_32_010_25.pt" https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt
aria2c -x 16 -s 16 -d $1 -o "ligandmpnn_v_32_020_25.pt" https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_020_25.pt
aria2c -x 16 -s 16 -d $1 -o "ligandmpnn_v_32_030_25.pt" https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_030_25.pt

#LigandMPNN with num_edges=32; atom_context_num=16
# aria2c -x 16 -s 16 -d $1 -o "ligandmpnn_v_32_005_16.pt" https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_005_16.pt
# aria2c -x 16 -s 16 -d $1 -o "ligandmpnn_v_32_010_16.pt" https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_16.pt
# aria2c -x 16 -s 16 -d $1 -o "ligandmpnn_v_32_020_16.pt" https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_020_16.pt
# aria2c -x 16 -s 16 -d $1 -o "ligandmpnn_v_32_030_16.pt" https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_030_16.pt

# aria2c -x 16 -s 16 -d $1 -o "publication_version_ligandmpnn_v_32_010_25.pt" https://files.ipd.uw.edu/pub/ligandmpnn/publication_version_ligandmpnn_v_32_010_25.pt

#Per residue label membrane ProteinMPNN
aria2c -x 16 -s 16 -d $1 -o "per_residue_label_membrane_mpnn_v_48_020.pt" https://files.ipd.uw.edu/pub/ligandmpnn/per_residue_label_membrane_mpnn_v_48_020.pt

#Global label membrane ProteinMPNN
aria2c -x 16 -s 16 -d $1 -o "global_label_membrane_mpnn_v_48_020.pt" https://files.ipd.uw.edu/pub/ligandmpnn/global_label_membrane_mpnn_v_48_020.pt

#SolubleMPNN
aria2c -x 16 -s 16 -d $1 -o "solublempnn_v_48_002.pt" https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_002.pt
aria2c -x 16 -s 16 -d $1 -o "solublempnn_v_48_010.pt" https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_010.pt
aria2c -x 16 -s 16 -d $1 -o "solublempnn_v_48_020.pt" https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt
aria2c -x 16 -s 16 -d $1 -o "solublempnn_v_48_030.pt" https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_030.pt

#LigandMPNN for side-chain packing (multi-step denoising model)
aria2c -x 16 -s 16 -d $1 -o "ligandmpnn_sc_v_32_002_16.pt" https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_sc_v_32_002_16.pt
