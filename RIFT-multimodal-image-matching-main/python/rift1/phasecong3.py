import numpy as np
from numpy.fft import fft2, ifft2, ifftshift


def lowpassfilter(size, cutoff, n):
    """生成低通滤波器"""
    rows, cols = size
    if cols % 2 == 1:
        x = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    else:
        x = np.arange(-cols / 2, cols / 2) / cols

    if rows % 2 == 1:
        y = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
    else:
        y = np.arange(-rows / 2, rows / 2) / rows

    x, y = np.meshgrid(x, y)
    radius = np.sqrt(x ** 2 + y ** 2)
    radius = ifftshift(radius)
    lp = 1 / (1 + (radius / cutoff) ** (2 * n))
    return lp


def phasecong3(im, nscale=4, norient=6, minWaveLength=3, mult=2.1,
               sigmaOnf=0.55, k=2.0, cutOff=0.5, g=10, noiseMethod=-1):
    """相位一致性计算"""
    im = im.astype(np.float64)
    if im.ndim == 3:
        im = np.mean(im, axis=2)  # 转为灰度图

    rows, cols = im.shape
    imagefft = fft2(im)
    zero = np.zeros((rows, cols))
    EO = [[np.zeros((rows, cols), dtype=np.complex128) for _ in range(norient)] for _ in range(nscale)]
    PC = [zero.copy() for _ in range(norient)]
    covx2 = zero.copy()
    covy2 = zero.copy()
    covxy = zero.copy()
    EnergyV = np.zeros((rows, cols, 3))
    pcSum = zero.copy()
    epsilon = 1e-4

    # 构建频率网格
    if cols % 2 == 1:
        xrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    else:
        xrange = np.arange(-cols / 2, cols / 2) / cols

    if rows % 2 == 1:
        yrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
    else:
        yrange = np.arange(-rows / 2, rows / 2) / rows

    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(-y, x)
    radius = ifftshift(radius)
    theta = ifftshift(theta)
    radius[0, 0] = 1  # 避免log(0)

    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # 低通滤波器
    lp = lowpassfilter((rows, cols), 0.45, 15)
    logGabor = []
    for s in range(nscale):
        wavelength = minWaveLength * (mult ** s)
        fo = 1.0 / wavelength
        lg = np.exp(-(np.log(radius / fo)) ** 2 / (2 * (np.log(sigmaOnf) ** 2)))
        lg *= lp
        lg[0, 0] = 0
        logGabor.append(lg)

    # 计算每个方向的相位一致性
    for o in range(norient):
        angl = o * np.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        dtheta = np.minimum(dtheta * norient / 2, np.pi)
        spread = (np.cos(dtheta) + 1) / 2

        sumE = zero.copy()
        sumO = zero.copy()
        sumAn = zero.copy()
        maxAn = zero.copy()

        for s in range(nscale):
            filt = logGabor[s] * spread
            eo = ifft2(imagefft * filt)
            EO[s][o] = eo
            An = np.abs(eo)
            sumAn += An
            sumE += np.real(eo)
            sumO += np.imag(eo)

            if s == 0:
                if noiseMethod == -1:
                    tau = np.median(sumAn) / np.sqrt(np.log(4))
                elif noiseMethod == -2:
                    hist, edges = np.histogram(sumAn, bins=50)
                    ind = np.argmax(hist)
                    rmode = (edges[ind] + edges[ind + 1]) / 2
                    tau = rmode
                maxAn = An
            else:
                maxAn = np.maximum(maxAn, An)

        # 能量计算
        EnergyV[..., 0] += sumE
        EnergyV[..., 1] += np.cos(angl) * sumO
        EnergyV[..., 2] += np.sin(angl) * sumO

        XEnergy = np.sqrt(sumE ** 2 + sumO ** 2) + epsilon
        MeanE = sumE / XEnergy
        MeanO = sumO / XEnergy

        Energy = zero.copy()
        for s in range(nscale):
            E = np.real(EO[s][o])
            O = np.imag(EO[s][o])
            Energy += E * MeanE + O * MeanO - np.abs(E * MeanO - O * MeanE)

        # 噪声阈值
        if noiseMethod >= 0:
            T = noiseMethod
        else:
            totalTau = tau * (1 - (1 / mult) ** nscale) / (1 - (1 / mult))
            EstMean = totalTau * np.sqrt(np.pi / 2)
            EstSigma = totalTau * np.sqrt((4 - np.pi) / 2)
            T = EstMean + k * EstSigma

        Energy = np.maximum(Energy - T, 0)
        width = (sumAn / (maxAn + epsilon) - 1) / (nscale - 1)
        weight = 1.0 / (1 + np.exp((cutOff - width) * g))
        PC[o] = weight * Energy / (sumAn + epsilon)
        pcSum += PC[o]

        # 协方差计算
        covx = PC[o] * np.cos(angl)
        covy = PC[o] * np.sin(angl)
        covx2 += covx ** 2
        covy2 += covy ** 2
        covxy += covx * covy

    # 最终计算
    covx2 /= (norient / 2)
    covy2 /= (norient / 2)
    covxy = 4 * covxy / norient
    denom = np.sqrt(covxy ** 2 + (covx2 - covy2) ** 2) + epsilon
    M = (covy2 + covx2 + denom) / 2
    m = (covy2 + covx2 - denom) / 2

    orien = np.degrees(np.arctan2(EnergyV[..., 2], EnergyV[..., 1]))
    orien[orien < 0] += 180
    featType = np.arctan2(EnergyV[..., 0], np.sqrt(EnergyV[..., 1] ** 2 + EnergyV[..., 2] ** 2))

    return M, m, orien, featType, PC, EO, T, pcSum