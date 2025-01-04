#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021 Hiroaki Santo
# https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

eps = 1e-12


def SchlickFresnel(u):
    m = torch.clamp(1.0 - u, 0, 1)
    return m**5


def GTR1(NdotH, a):
    # if a >= 1: return 1 / np.pi
    a2 = a**2
    t = 1 + (a2 - 1) * NdotH**2
    return (a2 - 1) / (np.pi * torch.log(a2) * t)


def GTR2(NdotH, a):
    a2 = torch.pow(a, 2)
    t = 1 + (a2 - 1) * NdotH**2
    denom = np.pi * t**2
    denom = torch.clamp(denom, min=eps)
    ret = a2 / denom
    return ret


def GTR2_aniso(NdotH, HdotX, HdotY, ax, ay):
    return 1 / (
        np.pi
        * ax
        * ay
        * ((HdotX / ax) ** 2 + (HdotY / ay) ** 2 + NdotH**2) ** 2
        + eps
    )


def smithG_GGX(NdotV, alphaG):
    a = alphaG**2
    b = NdotV**2
    return 1 / (NdotV + (a + b - a * b) ** 0.5 + eps)


def smithG_GGX_aniso(NdotV, VdotX, VdotY, ax, ay):
    return 1 / (
        NdotV
        + ((VdotX * ax) ** 2 + (VdotY * ay) ** 2 + NdotV**2) ** 0.5
        + eps
    )


def mon2lin(x):
    return x**2.2


def mix(x, y, a):
    return x * (1.0 - a) + y * a


def disney_bsdf(
    L: torch.Tensor,
    V: torch.Tensor,
    N: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    batch_num = len(L)
    # assert N.shape == (batch_num, 3), N.shape
    assert L.shape == (batch_num, 3), L.shape
    assert V.shape == (batch_num, 3), V.shape

    NdotL = torch.sum(N * L, dim=-1, keepdim=True)
    NdotV = torch.sum(N * V, dim=-1, keepdim=True)
    assert NdotL.shape == (batch_num, 1), NdotL.shape
    assert NdotV.shape == (batch_num, 1), NdotV.shape

    NdotL = torch.clamp(NdotL, min=0)
    NdotV = torch.clamp(NdotV, min=0)

    H = L + V
    H = F.normalize(H, p=2, dim=-1, eps=eps)
    NdotH = torch.sum(N * H, dim=-1, keepdim=True)
    LdotH = torch.sum(L * H, dim=-1, keepdim=True)
    assert H.shape == (batch_num, 3), H.shape

    X = torch.cross(
        torch.Tensor([[0, 1, 0]]).to(N.dtype).to(N.device).expand_as(L),
        N.reshape(-1, 3),
    )
    X = F.normalize(X, p=2, dim=-1, eps=eps)
    Y = torch.cross(N.reshape(-1, 3), X)  # (3,)
    Y = F.normalize(Y, p=2, dim=-1, eps=eps)

    assert X.shape == (batch_num, 3), X.shape
    assert Y.shape == (batch_num, 3), Y.shape

    baseColor = params["base_color"]
    metallic = params["metallic"]
    specular = params["specular"]
    roughness = params["roughness"]
    anisotropic = params["anisotropic"]
    sheen = params["sheen"]
    sheenTint = params["sheen_tint"]
    clearcoat = params["clearcoat"]
    clearcoatGloss = params["clearcoat_gloss"]
    specularTint = 0.0
    subsurface = 0.0
    assert baseColor.shape == (batch_num, 3), (baseColor.shape, batch_num)
    assert metallic.shape == (batch_num, 1), metallic.shape
    assert specular.shape == (batch_num, 1), specular.shape
    assert roughness.shape == (batch_num, 1), roughness.shape
    assert anisotropic.shape == (batch_num, 1), anisotropic.shape
    assert sheen.shape == (batch_num, 1), sheen.shape
    assert sheenTint.shape == (batch_num, 1), sheenTint.shape
    assert clearcoat.shape == (batch_num, 1), clearcoat.shape
    assert clearcoatGloss.shape == (batch_num, 1), clearcoatGloss.shape
    # baseColor = torch.clamp(baseColor, 0, 1.0)
    # metallic = torch.clamp(metallic, 0, 1.0)
    # specular = torch.clamp(specular, 0, 1.0)
    # roughness = torch.clamp(roughness, 0, 1.0)
    # anisotropic = torch.clamp(anisotropic, 0, 1.0)
    # sheen = torch.clamp(sheen, 0, 1.0)
    # sheenTint = torch.clamp(sheenTint, 0, 1.0)
    # clearcoat = torch.clamp(clearcoat, 0, 1.0)
    # clearcoatGloss = torch.clamp(clearcoatGloss, 0, 1.0)

    Cdlin = mon2lin(baseColor)
    Cdlum = 0.3 * Cdlin[:, 0] + 0.6 * Cdlin[:, 1] + 0.1 * Cdlin[:, 2]
    Cdlum = Cdlum[:, None]
    # Ctint = Cdlin / (Cdlum + eps)
    Ctint = torch.where(
        Cdlum > 0, Cdlin / (Cdlum + eps), torch.ones_like(Cdlum)
    )
    # vec3 Ctint = Cdlum > 0 ? Cdlin/Cdlum : vec3(1); // normalize lum. to isolate hue+sat

    assert Cdlin.shape == (batch_num, 3), Cdlin.shape
    assert Cdlum.shape == (batch_num, 1), Cdlum.shape
    assert Ctint.shape == (batch_num, 3), Ctint.shape

    Cspec0 = mix(
        specular * 0.08 * mix(torch.ones_like(Ctint), Ctint, specularTint),
        Cdlin,
        metallic,
    )
    Csheen = mix(torch.ones_like(Ctint), Ctint, sheenTint)
    assert Cspec0.shape == (batch_num, 3), Cspec0.shape
    assert Csheen.shape == (batch_num, 3), Csheen.shape

    # Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    # and mix in diffuse retro-reflection based on roughness
    FL = SchlickFresnel(NdotL)
    FV = SchlickFresnel(NdotV)
    assert roughness.shape == (batch_num, 1), roughness.shape
    assert LdotH.shape == (batch_num, 1), LdotH.shape
    Fd90 = 0.5 + 2.0 * LdotH * LdotH * roughness
    Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV)
    assert Fd90.shape == (batch_num, 1), Fd90.shape
    assert Fd.shape == (batch_num, 1), Fd.shape

    # Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    # 1.25 scale is used to (roughly) preserve albedo
    # Fss90 used to "flatten" retroreflection based on roughness
    Fss90 = LdotH * LdotH * roughness
    Fss = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV)
    ss = 1.25 * (Fss * (1.0 / (NdotL + NdotV + eps) - 0.5) + 0.5)
    assert ss.shape == (batch_num, 1), ss.shape

    # specular
    aspect = torch.sqrt(1 - anisotropic * 0.9) + eps
    ax = torch.clamp((roughness**2 / aspect), min=0.001)
    ay = torch.clamp((roughness**2 * aspect), min=0.001)
    assert ax.shape == (batch_num, 1), ax.shape
    assert ax.shape == ay.shape, (ax.shape, ay.shape)

    Ds = GTR2_aniso(
        NdotH,
        torch.sum(H * X, dim=-1, keepdim=True),
        torch.sum(H * Y, dim=-1, keepdim=True),
        ax,
        ay,
    )
    FH = SchlickFresnel(LdotH)
    Fs = mix(Cspec0, torch.ones_like(Cspec0), FH)
    Gs = smithG_GGX_aniso(
        NdotL,
        torch.sum(L * X, dim=-1, keepdim=True),
        torch.sum(L * Y, dim=-1, keepdim=True),
        ax,
        ay,
    )
    Gs = Gs * smithG_GGX_aniso(
        NdotV,
        torch.sum(V * X, dim=-1, keepdim=True),
        torch.sum(V * Y, dim=-1, keepdim=True),
        ax,
        ay,
    )

    # sheen
    Fsheen = FH * sheen * Csheen
    assert Fsheen.shape == (batch_num, 3), Fsheen.shape

    # clearcoat (ior = 1.5 -> F0 = 0.04)
    Dr = GTR1(NdotH, mix(0.1, 0.001, clearcoatGloss))
    Fr = mix(0.04, 1.0, FH)
    Gr = smithG_GGX(NdotL, 0.25) * smithG_GGX(NdotV, 0.25)
    assert Dr.shape == (batch_num, 1), Dr.shape
    assert Fr.shape == (batch_num, 1), Fr.shape
    assert Gr.shape == (batch_num, 1), Gr.shape

    ret1 = ((1 / np.pi) * mix(Fd, ss, subsurface) * Cdlin + Fsheen) * (
        1 - metallic
    )
    ret2 = Gs * Fs * Ds
    ret3 = 0.25 * clearcoat * Gr * Fr * Dr

    return ret1 + ret2 + ret3
