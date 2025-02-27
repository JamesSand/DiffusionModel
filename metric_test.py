import torch
import torch.distributions as dist

def gaussian_similarity_metrics(data_points, mu, cov):
    """
    计算数据点与高斯分布之间的相似度指标。
    
    Args:
        data_points (torch.Tensor): 数据点，形状为 (N, 2)。
        mu (torch.Tensor): 高斯分布的均值，形状为 (2,)。
        cov (torch.Tensor): 高斯分布的协方差矩阵，形状为 (2, 2)。
    
    Returns:
        dict: 包含以下指标的字典：
            - avg_log_likelihood: 平均对数似然（越高越好）
            - avg_mahalanobis_sq: 马氏距离平方的平均（接近2较好）
            - ks_statistic: KS统计量（越小越好）
            - prop_over_95th_chi2: 超过卡方95%分位数的比例（接近0.05较好）
    """
    # 计算数据点与均值的差值
    diff = data_points - mu.unsqueeze(0)  # (N, 2)
    
    # 处理奇异协方差矩阵，添加正则化项
    try:
        cov_inv = torch.linalg.inv(cov)
        log_det = torch.logdet(cov)
    except RuntimeError:
        eps = 1e-6 * torch.eye(cov.size(0), device=cov.device)
        cov_reg = cov + eps
        cov_inv = torch.linalg.inv(cov_reg)
        log_det = torch.logdet(cov_reg)
    
    # 计算马氏距离平方
    mahalanobis = torch.einsum('ni,ij,nj->n', diff, cov_inv, diff)
    
    # 计算平均对数似然
    d = data_points.size(1)
    const = -0.5 * d * torch.log(2 * torch.tensor(torch.pi, device=data_points.device))
    log_probs = const - 0.5 * log_det - 0.5 * mahalanobis
    avg_log_likelihood = log_probs.mean()
    
    # 平均马氏距离平方
    avg_mahalanobis_sq = mahalanobis.mean()
    
    # # 计算KS统计量（对比卡方分布）
    # df = d
    # chi2_dist = dist.Chi2(df=df)
    # mahalanobis_sorted = torch.sort(mahalanobis).values
    # n = mahalanobis.size(0)
    # empirical_cdf = torch.arange(1, n+1, device=mahalanobis.device) / n
    # theoretical_cdf = chi2_dist.cdf(mahalanobis_sorted)
    # ks_statistic = torch.max(torch.abs(empirical_cdf - theoretical_cdf))
    
    # # 计算超过卡方95%分位数的比例
    # threshold = chi2_dist.icdf(torch.tensor(0.95, device=mahalanobis.device))
    # prop_over_threshold = (mahalanobis > threshold).float().mean()
    
    return {
        'avg_log_likelihood': avg_log_likelihood.item(),
        'avg_mahalanobis_sq': avg_mahalanobis_sq.item(),
        # 'ks_statistic': ks_statistic.item(),
        # 'prop_over_95th_chi2': prop_over_threshold.item()
    }

# 示例用法
if __name__ == "__main__":
    # 生成样本数据（来自标准正态分布）
    mu = torch.tensor([0.0, 0.0])
    cov = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    data_points = torch.randn(1000, 2)  # 从目标分布采样
    
    metrics = gaussian_similarity_metrics(data_points, mu, cov)
    print("Similarity Metrics:")
    print(f"Average Log Likelihood: {metrics['avg_log_likelihood']:.4f}")
    print(f"Average Mahalanobis^2: {metrics['avg_mahalanobis_sq']:.4f} (Expected ≈ 2.0)")
    # print(f"KS Statistic: {metrics['ks_statistic']:.4f} (Closer to 0 is better)")
    # print(f"Proportion over 95th Chi2: {metrics['prop_over_95th_chi2']:.4f} (Expected ≈ 0.05)")