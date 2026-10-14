import requests
import os
import time

def fetch_small_novel_proteins(limit=50):
    print(f"🔍 正在向 PDB 数据库请求 {limit} 个最新解密的『微型单体蛋白』(150-400 aa)...")
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_accession_info.initial_release_date", "operator": "greater", "value": "2024-01-01T00:00:00Z"}},
                {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_protein", "operator": "equals", "value": 1}},
                {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_DNA", "operator": "equals", "value": 0}},
                {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_RNA", "operator": "equals", "value": 0}},
                # 黄金长度区间：显卡秒出结果！
                {"type": "terminal", "service": "text", "parameters": {"attribute": "entity_poly.rcsb_sample_sequence_length", "operator": "greater_or_equal", "value": 150}},
                {"type": "terminal", "service": "text", "parameters": {"attribute": "entity_poly.rcsb_sample_sequence_length", "operator": "less_or_equal", "value": 400}}
            ]
        },
        "request_options": {"paginate": {"start": 0, "rows": limit}, "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}]},
        "return_type": "entry"
    }

    response = requests.post(url, json=query)
    if response.status_code == 200:
        pdb_ids = [result['identifier'] for result in response.json().get('result_set', [])]
        print(f"✅ 成功锁定 {len(pdb_ids)} 个微型新蛋白！")
        return pdb_ids
    else:
        print(f"❌ API 请求失败: {response.status_code}")
        return []

def download_small_proteins(pdb_ids, base_dir="/data/wenhao/Protein-Hunter-server/DATA"):
    # 创建 pdb_protein 和 fasta 目录
    dirs = {
        "pdb": os.path.join(base_dir, "pdb_protein"), 
        "fasta": os.path.join(base_dir, "fasta")
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)
        
    print(f"\n📥 开始下载，保存在 {base_dir} ...")
    
    for idx, pdb_id in enumerate(pdb_ids):
        pdb_path = os.path.join(dirs["pdb"], f"{pdb_id}.pdb")
        fasta_path = os.path.join(dirs["fasta"], f"{pdb_id}.fasta")
        
        if os.path.exists(pdb_path) and os.path.exists(fasta_path):
            print(f"[{idx+1}/{len(pdb_ids)}] ⏩ {pdb_id} 已存在，跳过...")
            continue
            
        print(f"[{idx+1}/{len(pdb_ids)}] 正在下载 {pdb_id} ...")
        
        try:
            # 下载 PDB 文件 (注意: 部分新出的或超大结构可能只有 CIF，没有 PDB)
            pdb_res = requests.get(f"https://files.rcsb.org/view/{pdb_id}.pdb", timeout=30)
            if pdb_res.status_code == 200:
                with open(pdb_path, "w") as f: f.write(pdb_res.text)
            else:
                # 如果 .pdb 失败，尝试下载 .cif 备用
                print(f"    ⚠️ PDB {pdb_id} 不存在 (404)，尝试下载 CIF ...")
                cif_path = pdb_path.replace(".pdb", ".cif")
                cif_res = requests.get(f"https://files.rcsb.org/download/{pdb_id}.cif", timeout=30)
                if cif_res.status_code == 200:
                    with open(cif_path, "w") as f: f.write(cif_res.text)
                    print(f"    ✅ 已下载 CIF 替代 PDB")
                else:
                    print(f"    ❌ PDB/CIF {pdb_id} 均下载失败")
                    
            # 下载 FASTA 文件
            fasta_res = requests.get(f"https://www.rcsb.org/fasta/entry/{pdb_id}", timeout=30)
            if fasta_res.status_code == 200:
                with open(fasta_path, "w") as f: f.write(fasta_res.text)
            else:
                print(f"    ❌ FASTA {pdb_id} 下载失败: {fasta_res.status_code}")
                
            time.sleep(0.5) 
            
        except requests.exceptions.ReadTimeout:
            print(f"    ⚠️ {pdb_id} 下载超时！跳过...")
        except requests.exceptions.RequestException as e:
            print(f"    ❌ {pdb_id} 网络错误: {e}，跳过...")

    print("\n🎉 微型蛋白数据集下载完成！")

if __name__ == "__main__":
    ids = fetch_small_novel_proteins(limit=50)
    if ids: download_small_proteins(ids)
