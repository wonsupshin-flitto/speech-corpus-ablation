#!/usr/bin/env python3
"""부트스트랩·검정 공용 모듈 (논문1-1, 논문1 재집계).

왜 별도 모듈인가 (260827)
-------------------------
기존 집계 스크립트들이 직접 짠 선형합동법(LCG) 난수를 썼고, `s % n` 으로 인덱스를
뽑았다. LCG 의 **하위 비트는 주기가 짧아** n 이 2의 거듭제곱이면 같은 재표본만
반복된다. 실측: n=4·8·16·64 에서 재표본 1000회가 **전부 동일**, n=43 에서도 322종뿐.

증상은 "신뢰구간이 한 점으로 붕괴" 또는 "실제보다 좁은 CI" 다. 후자는 눈에 띄지
않아 그대로 논문에 들어갈 뻔했다.

여기서는 `random.Random`(메르센 트위스터)을 쓴다. seed 를 주므로 재현성은 같다.
"""
from __future__ import annotations

import math
import random
import statistics as st

NBOOT = 10000
SEED = 777

# t 분포 양측 95% 임계값. 표본이 작을 때 정규근사는 과신을 만든다.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 23: 2.069, 24: 2.064, 29: 2.045, 39: 2.023, 59: 2.001}


def t_crit(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.96


def bootstrap_ci(vals, nboot: int = NBOOT, seed: int = SEED):
    """평균과 백분위 부트스트랩 95% CI. (mean, lo, hi)"""
    v = list(vals)
    n = len(v)
    if n == 0:
        return (float("nan"),) * 3
    if n == 1:
        return (v[0], v[0], v[0])
    rng = random.Random(seed)
    means = []
    for _ in range(nboot):
        means.append(sum(v[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (st.mean(v), means[int(0.025 * nboot)], means[int(0.975 * nboot)])


def paired_t(vals):
    """단일표본(=대응차) t 검정. (mean, sd, t, df, ci_lo, ci_hi, significant)"""
    v = list(vals)
    n = len(v)
    if n < 2:
        return (st.mean(v) if v else float("nan"),) + (float("nan"),) * 6
    m, sd = st.mean(v), st.stdev(v)
    se = sd / math.sqrt(n)
    if se == 0:
        return (m, sd, float("inf"), n - 1, m, m, m != 0)
    t = m / se
    h = t_crit(n - 1) * se
    return (m, sd, t, n - 1, m - h, m + h, abs(t) > t_crit(n - 1))


def sign_test_p(pos: int, n: int) -> float:
    """부호검정 양측 p."""
    if n == 0:
        return float("nan")
    k = max(pos, n - pos)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


if __name__ == "__main__":
    # 자기검증: 깨진 LCG 와 달리 재표본이 실제로 다양한가
    for n in (4, 8, 16, 43, 64):
        v = [float(i) for i in range(n)]
        rng = random.Random(SEED)
        seen = {tuple(sorted(rng.randrange(n) for _ in range(n))) for _ in range(1000)}
        print(f"  n={n:>3}  서로 다른 재표본 {len(seen):>4}/1000 종")
