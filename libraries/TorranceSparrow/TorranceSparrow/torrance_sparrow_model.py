import torch

PI = 3.141592653589793

class TorranceSparrowParams:
    def __init__(
        self, kd: torch.Tensor, ks: torch.Tensor, sigma: torch.Tensor
    ) -> None:
        self.kd, self.ks, self.sigma = kd, ks, sigma
        self.kd = torch.clamp(self.kd, 0)
        self.ks = torch.clamp(self.ks, 0)
        self.sigma = torch.clamp(self.sigma, 0)
        if self.sigma.shape[-1] == 3:
            self.sigma = self.sigma.mean(dim=-1)
        if len(self.kd.shape) == 3:
            # print("Your input shape is (h,w,c). Automatically reshaped to (hxw,c).")
            h, w, _ = self.kd.shape
            self.kd = self.kd.reshape(h * w, -1)
            self.ks = self.ks.reshape(h * w, -1)
            self.sigma = self.sigma.reshape(h * w, -1)
        elif len(self.kd.shape) == 1:
            # print("Your input shape is (c). Automatically reshaped to (1,c).")
            self.kd = self.kd.reshape(1, -1)
            self.ks = self.ks.reshape(1, -1)
            self.sigma = self.sigma.reshape(1, -1)
        self.length = self.kd.shape[0]

    def __getitem__(self, index):
        return TorranceSparrowParams(
            kd=self.kd[index], ks=self.ks[index], sigma=self.sigma[index]
        )

    def __len__(self):
        return len(self.kd)


def tanTheta(x):
    temp = 1 - torch.square(x[..., 2])
    temp[temp < 0] = 0.0
    return torch.sqrt(temp) / x[..., 2]


def FresnelConductorExact(cosThetaI, eta, k):
    cosThetaI2 = cosThetaI * cosThetaI
    sinThetaI2 = 1 - cosThetaI2
    sinThetaI4 = sinThetaI2 * sinThetaI2

    temp1 = eta * eta - k * k - sinThetaI2
    a2pb2 = (temp1 * temp1 + 4 * k * k * eta * eta) ** 0.5
    a = (0.5 * (a2pb2 + temp1)) ** 0.5
    term1 = a2pb2 + cosThetaI2
    term2 = 2 * a * cosThetaI

    Rs2 = (term1 - term2) / (term1 + term2)

    term3 = a2pb2 * cosThetaI2 + sinThetaI4
    term4 = term2 * sinThetaI2

    Rp2 = Rs2 * (term3 - term4) / (term3 + term4)

    return 0.5 * (Rp2 + Rs2)


class MicrofacetDistribution:
    EPS = 1e-4

    def __init__(self, alpha):
        """
        alpha: roughness
        only support GGX
        """
        self.alpha = alpha
        self.alpha[self.alpha < self.EPS] = self.EPS

    def eval(self, m):
        # costheta２で割る操作は，あとでcostheta2をかけることでキャンセルされる
        beckmannExponent = torch.square(m[:, 0]) / torch.square(
            self.alpha[:, 0]
        ) + torch.square(m[:, 1]) / torch.square(self.alpha[:, 0])
        cos_theta_2 = m[..., 2] ** 2
        root = cos_theta_2 + beckmannExponent
        return 1 / (PI * torch.square(self.alpha[:, 0]) * torch.square(root))

    def G(self, wi, wo, m):
        return self.smithG1(wi, m) * self.smithG1(wo, m)

    def smithG1(self, v, m):
        zero_indices = torch.sum(v * m, dim=-1) * v[..., 2] <= 0
        tan = torch.abs(tanTheta(v))
        alpha = self.projectRoughness(v)
        root = alpha * tan
        result = 2.0 / (1.0 + torch.sqrt(1.0 + torch.square(root)))
        result[zero_indices] = 0.0
        return result

    def projectRoughness(self, v):
        invSinTheta2 = 1.0 / (
            1.0 - torch.clamp(torch.square(v[..., 2]), 0, 1 - self.EPS)
        )
        cosPhi2 = torch.square(v[..., 0]) * invSinTheta2
        sinPhi2 = torch.square(v[..., 1]) * invSinTheta2
        return torch.sqrt(
            cosPhi2 * torch.square(self.alpha[:, 0])
            + sinPhi2 * torch.square(self.alpha[:, 0])
        )


def torspa(kd, ks, sigma, wi, wo):
    assert isinstance(wi, torch.Tensor) and isinstance(wo, torch.Tensor)
    if not isinstance(kd, torch.Tensor):
        kd = torch.tensor(kd, dtype=torch.float32).to(wi.device)
        ks = torch.tensor(ks, dtype=torch.float32).to(wi.device)
        sigma = torch.tensor(sigma, dtype=torch.float32).to(wi.device)
    batch_num = wi.shape[0]
    if ks.shape[-1] == 3:
        return torspa3d(kd, ks, sigma, wi, wo)
    assert kd.shape == (batch_num, 3), f"{kd.shape}!={(batch_num,3)}"
    assert ks.shape == (batch_num, 1), f"{ks.shape}!={(batch_num,1)}"
    assert sigma.shape[0] == batch_num, f"{sigma.shape[0]}!={batch_num}"

    wh = (wi + wo) / torch.norm(wi + wo, dim=1, keepdim=True)
    distr = MicrofacetDistribution(sigma)

    G = distr.G(wi, wo, wh)
    D = distr.eval(wh)
    F = FresnelConductorExact(torch.sum(wo * wh, dim=-1), 0.0, 1.0)

    wiwo = wo[..., 2] * wi[..., 2]
    wiwo = torch.clamp(wiwo, distr.EPS, 1)
    return kd / PI + (ks.reshape(-1) * G * D * F / (4 * wiwo)).reshape(-1, 1)


def torspa3d(kd, ks, sigma, wi, wo):
    assert isinstance(wi, torch.Tensor) and isinstance(wo, torch.Tensor)
    if not isinstance(kd, torch.Tensor):
        kd = torch.tensor(kd, dtype=torch.float32).to(wi.device)
        ks = torch.tensor(ks, dtype=torch.float32).to(wi.device)
        sigma = torch.tensor(sigma, dtype=torch.float32).to(wi.device)
    batch_num = wi.shape[0]
    assert kd.shape == (batch_num, 3), f"{kd.shape}!={(batch_num,3)}"
    assert ks.shape == (batch_num, 3), f"{ks.shape}!={(batch_num,3)}"
    assert sigma.shape[0] == batch_num, f"{sigma.shape[0]}!={batch_num}"

    wh = (wi + wo) / torch.norm(wi + wo, dim=1, keepdim=True)
    distr = MicrofacetDistribution(sigma)

    G = distr.G(wi, wo, wh)
    D = distr.eval(wh)
    F = FresnelConductorExact(torch.sum(wo * wh, dim=-1), 0.0, 1.0)

    wiwo = wo[..., 2] * wi[..., 2]
    wiwo = torch.clamp(wiwo, distr.EPS, 1)

    rhos = G * D * F / (4 * wiwo)
    rhos = torch.tile(rhos.reshape(-1, 1), (1, 3))
    return kd / PI + (ks.reshape(-1, 3) * rhos).reshape(-1, 3)
