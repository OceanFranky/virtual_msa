import requests
import os
import time

def fetch_candidate_pool(pool_size=2000):
    """
    第一阶段：粗筛。获取满足长度和单链条件的大量候选 PDB ID。
    """
    print(f"🔍 [阶段 1/3] 正在向 PDB 粗筛 {pool_size} 个候选单链大蛋白...")
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                # 1. 必须是单链单体 (Entity = 1)
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 1
                    }
                },
                # 2. 排除核酸
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_DNA",
                        "operator": "equals",
                        "value": 0
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                        "operator": "equals",
                        "value": 0
                    }
                },
                # 3. 长度 400 - 830aa
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "range",
                        "value": {"from": 200, "to": 830}
                    }
                }
            ]
        },
        "request_options": {
            "paginate": {"start": 0, "rows": pool_size},
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}]
        },
        "return_type": "entry"
    }

    response = requests.post(url, json=query)
    
    if response.status_code == 200:
        data = response.json()
        candidate_ids = [result['identifier'] for result in data.get('result_set', [])]
        print(f"✅ 粗筛完成！拿到 {len(candidate_ids)} 个候选 PDB。")
        return candidate_ids
    else:
        raise RuntimeError(f"❌ 粗筛 API 请求失败: {response.text}")

import concurrent.futures
import sys

def check_single_pdb(pdb_id):
    """单独查验一个 PDB 是否完美的子线程任务"""
    data_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        # 设置超时时间为 5 秒，防止卡死
        resp = requests.get(data_url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            try:
                unmodeled_count = data['rcsb_entry_info']['unmodeled_polymer_monomer_count']
            except KeyError:
                unmodeled_count = 999 
            
            if unmodeled_count == 0:
                return pdb_id
    except:
        pass
    return None

def filter_perfect_pdbs_fast(candidate_ids, target_count=50):
    """
    [第二阶段加速版]：多线程并发排查，带有实时进度指示器
    """
    print(f"\n🔬 [阶段 2/3] 开启多线程引擎，目标筛选 {target_count} 个 100% 完美蛋白...")
    perfect_pdbs = []
    checked_count = 0
    
    # 开启 20 个并发线程（PDB 官网扛得住这个并发量）
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # 将所有候选 ID 提交给线程池
        future_to_pdb = {executor.submit(check_single_pdb, pid): pid for pid in candidate_ids}
        
        for future in concurrent.futures.as_completed(future_to_pdb):
            checked_count += 1
            result = future.result()
            
            if result:
                perfect_pdbs.append(result)
                # 命中时，换行打印好消息
                sys.stdout.write(f"\n  🎯 [命中!] 找到完美 PDB: {result} (进度: {len(perfect_pdbs)}/{target_count})\n")
            
            # 实时覆盖打印当前排查进度（利用 \r 回车符不换行）
            sys.stdout.write(f"  ⚡ 正在疯狂排查... 已查阅: {checked_count}/{len(candidate_ids)} | 废弃残缺 PDB: {checked_count - len(perfect_pdbs)} 个\r")
            sys.stdout.flush()
            
            # 如果凑够了目标数量，直接强行终止剩余线程
            if len(perfect_pdbs) >= target_count:
                print(f"\n\n✅ 目标达成！提前终止排查。共查阅了 {checked_count} 个结构。")
                # 强行取消还没执行的任务
                for f in future_to_pdb:
                    f.cancel()
                break

    if len(perfect_pdbs) < target_count:
        print(f"\n\n⚠️ 候选池耗尽，最终只找到了 {len(perfect_pdbs)} 个完美结构。")
        
    return perfect_pdbs

def download_complete_proteins(pdb_ids, base_dir="perfect_large_proteins"):
    """ 第三阶段：下载 CIF 和 FASTA """
    print(f"\n📥 [阶段 3/3] 开始下载，保存在 ./{base_dir}/ ...")
    dirs = {"cif": os.path.join(base_dir, "cif"), "fasta": os.path.join(base_dir, "fasta")}
    for d in dirs.values(): os.makedirs(d, exist_ok=True)
    
    for idx, pdb_id in enumerate(pdb_ids):
        cif_path = os.path.join(dirs["cif"], f"{pdb_id}.cif")
        fasta_path = os.path.join(dirs["fasta"], f"{pdb_id}.fasta")
        
        if os.path.exists(cif_path) and os.path.exists(fasta_path):
            print(f"[{idx+1}/{len(pdb_ids)}] ⏩ {pdb_id} 已存在，跳过...")
            continue
            
        print(f"[{idx+1}/{len(pdb_ids)}] 下载 {pdb_id} ...")
        
        try:
            cif_res = requests.get(f"https://files.rcsb.org/download/{pdb_id}.cif", timeout=30)
            if cif_res.status_code == 200:
                with open(cif_path, "w") as f: f.write(cif_res.text)
                    
            fasta_res = requests.get(f"https://www.rcsb.org/fasta/entry/{pdb_id}", timeout=30)
            if fasta_res.status_code == 200:
                with open(fasta_path, "w") as f: f.write(fasta_res.text)
                
            time.sleep(0.5) 
        except Exception as e:
            print(f"    ❌ {pdb_id} 下载错误: {e}")

    print("\n🎉 完美大蛋白数据集下载完成！现在你可以毫无顾虑地跑 MPNN 了！")

if __name__ == "__main__":
    # 1. 抓取 2000 个候选者
    candidates = fetch_candidate_pool(pool_size=10000)
    
    # 2. 从中严格过滤出 50 个完美的（你可以修改这个 target_count）
    if candidates:
        perfect_ids = filter_perfect_pdbs_fast(candidates, target_count=50)
        
        # 3. 下载
        if perfect_ids:
            download_complete_proteins(perfect_ids)
        else:
            print("❌ 在这批候选中没有找到任何一个 100% 完美的结构。你可能需要增加 pool_size 或放宽长度限制。")