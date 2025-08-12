# scan_alpha.py
import numpy as np
import pandas as pd

# === 1) 你已有：高精度 A(α) 计算器（我们前面写的那个） ===
from test_A import compute_A_single  # 确保导入路径正确

# === 2) 需要你提供/替换：一个“计算 log-likelihood(α) 不带 A”的函数 ===
# 请把下面这个 stub 改成你项目里实际的似然评估入口（并确保 A 被关掉/设为1）
def loglike_without_A(alpha, fixed_mu_DM=12.91, **kwargs) -> float:
    """
    返回在 alpha 下的 log L（不乘 A）。
    你需要：
      - 固定 mu_DM（和你要测试的其他超参，通常用真值）
      - 在似然评估处把 A_eta=1（或直接跳过 A）
    """
    # ====== TODO: 用你自己的接口替换下面这行 ======
    # e.g. from likelihood import log_likelihood
    # return log_likelihood(mu_DM=fixed_mu_DM, alpha=alpha, use_A=False, ...)
    raise NotImplementedError("把这段替换为你实际的似然评估函数（A=1）")

# === 3) 主程序：扫描 α 并输出三条曲线 ===
def scan_alpha(
    alphas=None,
    mu_DM=12.91,
    # A(α) 的计算精度参数：
    n_samples_A=200_000,
    ms_points=400,
    m_lim=26.5,
    zl=0.3,
    zs=2.0,
    n_jobs=8,
):
    if alphas is None:
        alphas = np.linspace(0.06, 0.22, 9)  # 自己改区间/步长

    logL_list = []
    A_list = []

    # 先算似然（不带 A）
    for a in alphas:
        ll = loglike_without_A(a, fixed_mu_DM=mu_DM)
        logL_list.append(ll)

    # 再算 A(α)（高精度）
    Avals = compute_A_single(
        mu_DM=mu_DM,
        alpha_list=list(alphas),
        n_samples=n_samples_A,
        ms_points=ms_points,
        m_lim=m_lim,
        zl=zl, zs=zs,
        n_jobs=n_jobs
    )
    for a in alphas:
        A_list.append(Avals[a])

    logL = np.array(logL_list, dtype=float)
    A = np.array(A_list, dtype=float)
    neglogA = -np.log(A)
    post = logL - np.log(A)

    df = pd.DataFrame({
        "alpha": alphas,
        "logL_noA": logL,
        "-logA": neglogA,
        "logPosterior": post
    })
    return df

if __name__ == "__main__":
    df = scan_alpha()
    print(df.to_string(index=False))
    # 你也可以 df.to_csv("scan_alpha.csv", index=False)
