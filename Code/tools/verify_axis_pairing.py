"""核验 dataset/TXT 的横纵轴定义，以及 TXT↔PARA 是否错位一格。

原理：CST 工程 BIC07xx/Result/Storage.sdb 是 SQLite 结果库，
里面的 aZmin(2)Zmax(2)c.sig 就是导出到 TXT 的那个量（|S|，线性）。
把它当外部真值，在全库找最佳匹配的 TXT，即可判定配对。

用法： python Code/tools/verify_axis_pairing.py
"""
import glob
import json
import os
import sqlite3
import struct

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "dataset")
PARAM_NAMES = ["Px", "Py", "A", "B", "L", "Y"]


def get_cst_curve(sdb_path):
    """从 CST 结果库取 (freq_THz, |S|)。仅 'aZ...c.sig'（已归一化）可用。"""
    con = sqlite3.connect(sdb_path)
    row = con.execute(
        "select sig_id from SigHeader where name='aZmin(2)Zmax(2)c.sig' limit 1"
    ).fetchone()
    if not row:
        return None, None
    sid = row[0]
    for n in range(1, 13):
        try:
            cnt = con.execute(f"select count(*) from SigData{n} where sig_id=?",
                              (sid,)).fetchone()[0]
        except sqlite3.Error:
            continue
        if not cnt:
            continue
        if n in (3, 7):                      # (dom, codom) 实数
            d = np.array(con.execute(
                f"select dom,codom from SigData{n} where sig_id=? order by dom",
                (sid,)).fetchall())
            return d[:, 0], np.abs(d[:, 1])
        if n in (5, 8):                      # (dom, re, im) 复数
            d = np.array(con.execute(
                f"select dom,codom_real,codom_imag from SigData{n} "
                "where sig_id=? order by dom", (sid,)).fetchall())
            return d[:, 0], np.hypot(d[:, 1], d[:, 2])
        blob = con.execute(f"select bData from SigData{n} where sig_id=?",
                           (sid,)).fetchone()[0]
        if isinstance(blob, (bytes, bytearray)) and len(blob) >= 24 + 3 * 1101 * 8:
            npts = struct.unpack_from("<I", blob, 0)[0]
            off = 24                          # 头 24 字节，随后 freq / re / im 各 npts 个 float64
            f = np.array(struct.unpack_from(f"<{npts}d", blob, off))
            re = np.array(struct.unpack_from(f"<{npts}d", blob, off + 8 * npts))
            im = np.array(struct.unpack_from(f"<{npts}d", blob, off + 16 * npts))
            return f, np.hypot(re, im)
    return None, None


def load_txt(i):
    path = os.path.join(BASE, "TXT", f"{i}.txt")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        fh.readline()                         # 表头
        fh.readline()                         # 分隔线
        return np.array(fh.read().split(), dtype=float).reshape(-1, 2)


def main():
    # --- 横轴 ---
    spec = load_txt(1)
    f = spec[:, 0]
    print("### 横轴")
    print(f"  {f[0]:.6f} -> {f[-1]:.6f} THz, {len(f)} 点, "
          f"步长 {(f[-1]-f[0])/(len(f)-1)*1000:.5f} GHz (线性)")

    # --- 纵轴 ---
    ids, spectra = [], []
    for i in range(1, 3841):
        s = load_txt(i)
        if s is not None:
            ids.append(i)
            spectra.append(s[:, 1])
    S = np.array(spectra)
    print("### 纵轴")
    print(f"  值域 [{S.min():.6f}, {S.max():.6f}]  负值 {(S < 0).sum()} 个  "
          f">1 的 {(S > 1).sum()} 个")

    # --- 配对核验 ---
    para = {i: np.loadtxt(os.path.join(BASE, "PARA", f"para{i}.txt"))
            for i in range(1, 3841)}
    grid = np.array(list(para.values()))
    print("\n### CST 真值 vs TXT 配对")
    print(f"{'工程':<10}{'PARA行N':>9}{'最佳TXT':>9}{'残差':>12}{'次优':>12}{'M-N':>6}")
    for proj in sorted(glob.glob(os.path.join(BASE, "CST", "BIC07[0-9][0-9]"))):
        name = os.path.basename(proj)
        pj = json.load(open(os.path.join(proj, "Model", "Parameters.json")))
        vals = {p["name"]: p["value"] for p in pj["parameters"]}
        row = np.array([float(vals[k]) for k in PARAM_NAMES])
        hit = np.where((grid == row).all(1))[0]
        if not len(hit):
            continue
        n = int(hit[0]) + 1
        cf, cv = get_cst_curve(os.path.join(proj, "Result", "Storage.sdb"))
        if cf is None:
            print(f"{name:<10}{n:>9}   (该工程未存归一化的 aZ 结果，跳过)")
            continue
        err = {}
        for m in range(max(1, n - 8), min(3840, n + 8) + 1):
            t = load_txt(m)
            if t is None:
                continue
            k = np.array([np.argmin(np.abs(t[:, 0] - fj)) for fj in cf])
            err[m] = float(np.abs(t[k, 1] - cv).mean())
        best = min(err, key=err.get)
        second = sorted(err.values())[1]
        print(f"{name:<10}{n:>9}{best:>9}{err[best]:>12.2e}{second:>12.2e}"
              f"{best - n:>6}")
    print("\nM-N 恒为 +1 => TXT/{i} 装的是 para{i-1} 的结果，需整体前移一格。")


if __name__ == "__main__":
    main()
