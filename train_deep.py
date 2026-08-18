import torch, math
from pathlib import Path
from torch import optim
from apinn.models import MLP_APINN
from apinn.physics import (equilibrium_residuals, constitutive_increment, traction_residual_on_circle, 
                           symmetry_residuals, lame_parameters)
from apinn.sampling import (sample_quarter_annulus, sample_on_circle, sample_on_axis, importance_resample)

torch.set_default_dtype(torch.float64)
THIS_DIR = Path(__file__).resolve().parent

def to_rad(deg): return deg * math.pi / 180.0


class DeepBuriedAPINN:
    """
        深埋隧道 APINN 模型封装：
        - 保存材料参数、几何参数、荷载、初始应力
        - 包含一个神经网络 MLP_APINN
        - 提供前向拆分、损失计算、训练流程等方法
    """
    def __init__(self):
        # ==================== 严格按照论文第5.1节参数设置 ====================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

        # 材料参数（表1）
        self.E = 10.0        # 弹性模量，数值意义上=10 (可视作 10 MPa 的无量纲化)
        self.nu = 0.2          # 泊松比
        self.c = 20.0        # 黏聚力，数值意义上=20 (可视作 20 kPa / 1 kPa 之类)
        self.phi = to_rad(30.0)   # 内摩擦角 30°
        self.psi = to_rad(30.0)   # 剪胀角 30°（关联流动）

        # 几何与荷载（第5.1节原文）
        self.a = 1.0          # 隧道半径 1 m
        self.R = 3.0          # 计算域外边界半径 3 m（原文明确写 R = 3 m！）
        self.p0 = 10.0       # 远场静水压力，数值意义上=10 (MPa)

        # 初始地应力（张力为正，压缩为负）
        p0_tensor = torch.tensor(-self.p0, device=self.device, dtype=torch.float64)
        self.r0 = (p0_tensor, p0_tensor, p0_tensor)   # σ_xx^0 = σ_yy^0 = σ_zz^0 = -10 MPa

        # 神经网络结构（原文：6层隐藏层，每层40个神经元，Tanh激活）
        self.model = MLP_APINN(hidden=40, depth=6).to(self.device)

    def forward_split(self, X):
        out = self.model(X)
        u = out[..., 0:2]     # 位移 u_x, u_y
        s = out[..., 2:]      # 应力 σ_xx, σ_yy, σ_zz, σ_xy
        return u, s

    def loss_terms(self, X_res, X_b_in, X_b_out, X_symx, X_symy, p_in):
        """
                计算当前采样点集下的各项损失：
                输入:
                    X_res   : 域内残差点（四分之一圆环）
                    X_b_in  : 内边界点（隧道内壁）
                    X_b_out : 外边界点（远场人工边界）
                    X_symx  : x 轴对称边界上的点
                    X_symy  : y 轴对称边界上的点
                    p_in    : 隧道内支撑压力（正号表示“压力大小”，在本代码中会转换为 -p_in）

                输出:
                    L_E : 动量平衡 PDE 残差损失
                    L_C : 本构增量残差损失
                    L_B : 边界（内压 + 外压 + 对称）损失
                    L_Y : 屈服函数惩罚损失 (F>0 的区域)
                    stats: dict 形式的标量信息，方便打印
        """
        u_res, s_res = self.forward_split(X_res)
        E_res = equilibrium_residuals(s_res, X_res)
        C_res, F_res, dk = constitutive_increment(u_res, s_res, X_res, self.E, self.nu, self.c, self.phi, self.r0)

        u_in, s_in = self.forward_split(X_b_in)
        # 关键修改2：内边界支持压力为 -p_in（张力为正号制）
        t_in = traction_residual_on_circle(s_in, X_b_in, self.a, target_pressure=-p_in)

        u_out, s_out = self.forward_split(X_b_out)
        # 关键修改3：外边界远场静水压力为 -p0
        t_out = traction_residual_on_circle(s_out, X_b_out, self.R, target_pressure=-self.p0)

        u_sx, s_sx = self.forward_split(X_symx)
        symx = symmetry_residuals(s_sx, u_sx, X_symx, axis='x')

        u_sy, s_sy = self.forward_split(X_symy)
        symy = symmetry_residuals(s_sy, u_sy, X_symy, axis='y')

        Fy_penalty = torch.relu(F_res)[..., None]

        wE, wC, wB, wY = 1.0, 1.0, 1.0, 1.0
        L_E = (E_res**2).mean()
        L_C = (C_res**2).mean()
        L_B = (t_in**2).mean() + (t_out**2).mean() + (symx**2).mean() + (symy**2).mean()
        L_Y = (Fy_penalty**2).mean()
        return L_E, L_C, L_B, L_Y, {'E':L_E.item(), 'C':L_C.item(), 'B':L_B.item(), 'Y':L_Y.item()}

    def train_stage(self):
        """
                按论文第 5.1 节“精确采样策略”进行训练：
                - 初始：域内 300 点 + 边界 50 点
                - 先进行 10000 步 Adam 预训练
                - 然后进行 10 轮自适应采样：
                    每轮在 4000 个点上评估残差 → 选 20 个残差大的点加入 X_res
                    然后跑 1000 步 Adam + 1000 步 L-BFGS
        """
        device = self.device
        dtype = torch.float64

        # ==================== 第5.1节原文精确采样策略 ====================
        n_init_res = 300  # 初始域内配点 300 个
        n_b = 50  # 边界点总数 50 个
        eval_n = 4000  # 每轮评估用 4000 个点
        adapt_new_each = 20  # 每轮根据残差最大处添加 20 个新点
        adapt_rounds = 10  # 总共 10 轮自适应

        # 初始采样
        X_res = sample_quarter_annulus(n_init_res, self.a, self.R, device, dtype)
        X_b_in = sample_on_circle(n_b // 4, self.a, device, dtype)
        X_b_out = sample_on_circle(n_b // 4, self.R, device, dtype)
        X_symx = sample_on_axis(n_b // 4, 'x', self.R, device, dtype)
        X_symy = sample_on_axis(n_b // 4, 'y', self.R, device, dtype)

        # 优化器
        optimizer_adam = optim.Adam(self.model.parameters(), lr=1e-3)
        optimizer_lbfgs = optim.LBFGS(self.model.parameters(), history_size=50, max_iter=1000,
                                      line_search_fn="strong_wolfe")
        p_in = 8.0           # 内支撑压力数值 8 (MPa)

        def closure():
            """
            L-BFGS 需要的 closure 函数：
            每一次调用都要：
            1) 清零梯度
            2) 重新计算损失
            3) 反向传播梯度
            4) 返回标量损失
            """
            optimizer_lbfgs.zero_grad()
            L_E, L_C, L_B, L_Y, stats = self.loss_terms(X_res, X_b_in, X_b_out, X_symx, X_symy, p_in)
            L = L_E + L_C + L_B + L_Y
            L.backward()
            return L

        # ==================== 初始大Adam训练（原文10000步） ====================
        print("开始初始Adam训练（10000步）...")
        for i in range(10000):
            optimizer_adam.zero_grad()
            L_E, L_C, L_B, L_Y, _ = self.loss_terms(X_res, X_b_in, X_b_out, X_symx, X_symy, p_in)
            L = L_E + L_C + L_B + L_Y
            L.backward()
            optimizer_adam.step()
            if (i + 1) % 2000 == 0:
                print(f"Step {i + 1}/10000, Total Loss: {L.item():.2e}")

        # ==================== 10轮自适应残差采样（原文精确流程） ====================
        for r in range(adapt_rounds):
            print(f"\n自适应轮次 {r + 1}/{adapt_rounds}，当前配点数：{X_res.shape[0]}")

            # 在4000个均匀点上评估残差
            X_eval = sample_quarter_annulus(eval_n, self.a, self.R, device, dtype)
            u_ev, s_ev = self.forward_split(X_eval)
            E_ev = equilibrium_residuals(s_ev, X_eval)
            C_ev, _, _ = constitutive_increment(u_ev, s_ev, X_eval, self.E, self.nu, self.c, self.phi, self.r0)
            # 计算综合残差模长：把 PDE 残差和本构残差合在一起
            res_mag = torch.sqrt((E_ev ** 2).sum(-1, keepdim=True) + (C_ev ** 2).sum(-1, keepdim=True))

            # 添加残差最大的20个点从 4000 个候选点中，按残差大小的概率分布选出 adapt_new_each=20 个新点
            X_new = importance_resample(X_eval.detach(), res_mag.detach(), k=1.0, m=0.05, n_new=adapt_new_each)
            X_new.requires_grad_(True)
            X_res = torch.cat([X_res, X_new], dim=0)

            # 每轮：1000步 Adam + 1 次 L-BFGS(step 内部最多迭代 1000 次)
            # 先用 Adam 在当前自适应采样点上微调 1000 步
            for _ in range(1000):
                optimizer_adam.zero_grad()
                L_E, L_C, L_B, L_Y, _ = self.loss_terms(
                    X_res, X_b_in, X_b_out, X_symx, X_symy, p_in
                )
                L = L_E + L_C + L_B + L_Y
                L.backward()
                optimizer_adam.step()

            # 再交给 L-BFGS，用 closure 重新计算一次损失和梯度
            optimizer_lbfgs.step(closure)

        # ========== 新增：训练结束后做一次总损失检查 ==========
        # ========== 训练结束后做一次总损失检查（不要用 no_grad，否则 physics 里的 autograd.grad 会报错） ==========
        L_E, L_C, L_B, L_Y, _ = self.loss_terms(
            X_res, X_b_in, X_b_out, X_symx, X_symy, p_in
        )
        L = L_E + L_C + L_B + L_Y
        print(f"\nFinal loss: {L.item():.3e}")
        print(
            "  L_E = %.3e, L_C = %.3e, L_B = %.3e, L_Y = %.3e"
            % (L_E.item(), L_C.item(), L_B.item(), L_Y.item())
        )
        # ========== 调试结束 ==========

        print("训练完成！已严格按照论文第5.1节流程训练。")
        checkpoint = THIS_DIR / "checkpoints" / "apinn_deep_elastic_paper_exact.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint)
        print(f"模型参数已保存：{checkpoint}")

def main():
    print("开始复现论文《Acta Geotechnica 2025》深埋隧道弹性阶段（Fig.10）\n")
    model = DeepBuriedAPINN()
    model.train_stage()

if __name__ == "__main__":
    main()
