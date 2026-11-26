#!/usr/bin/env python3
"""Differential reference verifier for keel::linalg against NumPy, SciPy, and MATLAB R2025b."""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import scipy.linalg as sla
from scipy.optimize import nnls


def load_mat_f64(path: str, m: int, n: int) -> np.ndarray:
    return np.fromfile(path, dtype="<f8").reshape((m, n), order="F")


def load_mat_c64(path: str, m: int, n: int) -> np.ndarray:
    return np.fromfile(path, dtype="<c16").reshape((m, n), order="F")


def load_vec_f64(path: str) -> np.ndarray:
    return np.fromfile(path, dtype="<f8")


def load_vec_c64(path: str) -> np.ndarray:
    return np.fromfile(path, dtype="<c16")


def assert_close(name: str, got: np.ndarray, ref: np.ndarray, tol: float, results: list):
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    err = float(np.max(np.abs(got - ref)))
    scale = max(1.0, float(np.max(np.abs(ref))))
    rel = err / scale
    if not np.isfinite(rel) or rel > tol:
        raise AssertionError(f"{name}: max_abs={err:.3e}, rel={rel:.3e} > tol={tol:.3e}")
    results.append((name, f"rel={rel:.2e}"))


def sort_complex(v: np.ndarray) -> np.ndarray:
    idx = np.lexsort((np.round(v.imag, 8), np.round(v.real, 8)))
    return v[idx]


def run_matlab_check(dump_dir: str, results: list):
    matlab_exe = shutil.which("matlab")
    if not matlab_exe:
        cand = r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
        if os.path.exists(cand):
            matlab_exe = cand
    if not matlab_exe:
        return

    d = dump_dir.replace("\\", "/")
    out_m = f"{d}/matlab_linalg_ref.bin"
    # Verify high-norm expm, complex logm/sqrtm, lsqnonneg, and gsvd against MATLAB R2025b.
    m_code = (
        f"f=fopen('{d}/expm_Ar.bin','rb'); Ar=reshape(fread(f,'double'),6,6); fclose(f); "
        "E_Ar = expm(Ar); "
        f"f=fopen('{d}/mf_A.bin','rb'); v=fread(f,'double'); fclose(f); "
        "Ac = reshape(complex(v(1:2:end),v(2:2:end)),6,6); "
        "L_Ac = logm(Ac); S_Ac = sqrtm(Ac); "
        f"f=fopen('{d}/nnls_C.bin','rb'); NC=reshape(fread(f,'double'),10,5); fclose(f); "
        f"f=fopen('{d}/nnls_d.bin','rb'); nd=fread(f,'double'); fclose(f); "
        "nx = lsqnonneg(NC, nd); "
        f"f=fopen('{d}/gsvd_A.bin','rb'); GA=reshape(fread(f,'double'),4,3); fclose(f); "
        f"f=fopen('{d}/gsvd_B.bin','rb'); GB=reshape(fread(f,'double'),3,3); fclose(f); "
        "gs = sort(gsvd(GA, GB)); "
        "pack = [E_Ar(:); real(L_Ac(:)); imag(L_Ac(:)); real(S_Ac(:)); imag(S_Ac(:)); nx(:); gs(:)]; "
        f"f=fopen('{out_m}','wb'); fwrite(f, pack, 'double'); fclose(f);"
    )
    proc = subprocess.run([matlab_exe, "-batch", m_code], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"MATLAB reference failed:\n{proc.stdout}\n{proc.stderr}")

    raw = load_vec_f64(os.path.join(dump_dir, "matlab_linalg_ref.bin"))
    off = 0
    m_expm = raw[off:off + 36].reshape((6, 6), order="F"); off += 36
    m_logm = (raw[off:off + 36] + 1j * raw[off + 36:off + 72]).reshape((6, 6), order="F"); off += 72
    m_sqrtm = (raw[off:off + 36] + 1j * raw[off + 36:off + 72]).reshape((6, 6), order="F"); off += 72
    m_nnls = raw[off:off + 5]; off += 5
    m_gsvd = raw[off:off + 3]

    assert_close("matlab:expm_high_norm", load_mat_f64(os.path.join(dump_dir, "expm_Ar_out.bin"), 6, 6), m_expm, 1e-11, results)
    assert_close("matlab:logm_c64", load_mat_c64(os.path.join(dump_dir, "logm_Ac_out.bin"), 6, 6), m_logm, 1e-11, results)
    assert_close("matlab:sqrtm_c64", load_mat_c64(os.path.join(dump_dir, "sqrtm_Ac_out.bin"), 6, 6), m_sqrtm, 1e-11, results)
    assert_close("matlab:lsqnonneg", load_vec_f64(os.path.join(dump_dir, "nnls_x.bin")), m_nnls, 1e-10, results)
    assert_close("matlab:gsvd_values", load_vec_f64(os.path.join(dump_dir, "gsvd_sig.bin")), m_gsvd, 1e-10, results)


def main():
    if len(sys.argv) < 2:
        print("Usage: verify_reference.py <path/to/test_linalg.exe>", file=sys.stderr)
        sys.exit(2)
    exe = sys.argv[1]
    results = []

    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([exe, "--dump-ref", tmp], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"test_linalg --dump-ref failed:\n{proc.stdout}\n{proc.stderr}")

        # 1. GEMM
        A = load_mat_f64(os.path.join(tmp, "gemm_A.bin"), 73, 59)
        B = load_mat_f64(os.path.join(tmp, "gemm_B.bin"), 59, 67)
        C = load_mat_f64(os.path.join(tmp, "gemm_C.bin"), 73, 67)
        assert_close("numpy:gemm_73x67x59", C, 1.3 * (A @ B), 1e-12, results)

        # 2. Linear solve, complex inv, tall least squares
        sA = load_mat_f64(os.path.join(tmp, "solve_A.bin"), 140, 140)
        sB = load_mat_f64(os.path.join(tmp, "solve_B.bin"), 140, 4)
        sX = load_mat_f64(os.path.join(tmp, "solve_X.bin"), 140, 4)
        assert_close("scipy:la_solve_140x140", sX, sla.solve(sA, sB), 1e-11, results)

        iAc = load_mat_c64(os.path.join(tmp, "inv_Ac.bin"), 18, 18)
        iAc_out = load_mat_c64(os.path.join(tmp, "inv_Ac_out.bin"), 18, 18)
        assert_close("scipy:matrix_inv_c64_18x18", iAc_out, sla.inv(iAc), 1e-11, results)

        lA = load_mat_f64(os.path.join(tmp, "lstsq_A.bin"), 24, 6)
        lb = load_vec_f64(os.path.join(tmp, "lstsq_b.bin"))
        lx = load_vec_f64(os.path.join(tmp, "lstsq_x.bin"))
        assert_close("scipy:la_solve_tall_qr", lx, sla.lstsq(lA, lb)[0], 1e-11, results)

        # 3. Cholesky, SVD, Symmetric & Generalized Eigensolvers
        cA = load_mat_f64(os.path.join(tmp, "chol_A.bin"), 90, 90)
        cR = load_mat_f64(os.path.join(tmp, "chol_R.bin"), 90, 90)
        assert_close("scipy:cholesky_upper_90x90", cR, sla.cholesky(cA, lower=False), 1e-11, results)

        svdA = load_mat_f64(os.path.join(tmp, "svd_A.bin"), 12, 7)
        svdS = load_vec_f64(os.path.join(tmp, "svd_S.bin"))
        assert_close("scipy:svd_singular_values", svdS, sla.svdvals(svdA), 1e-11, results)

        eigA = load_mat_f64(os.path.join(tmp, "eig_A.bin"), 24, 24)
        eigD = load_vec_f64(os.path.join(tmp, "eig_D.bin"))
        assert_close("scipy:eig_sym_24x24", eigD, sla.eigvalsh(eigA), 1e-11, results)

        geA = load_mat_f64(os.path.join(tmp, "geig_A.bin"), 10, 10)
        geB = load_mat_f64(os.path.join(tmp, "geig_B.bin"), 10, 10)
        geD = load_vec_f64(os.path.join(tmp, "geig_D.bin"))
        assert_close("scipy:eig_sym_generalized", geD, sla.eigvalsh(geA, geB), 1e-11, results)

        # 4. Schur eigenvalues, QZ eigenvalues, expm, logm, sqrtm, funm, sylvester, expmv, nnls
        mfA = load_mat_c64(os.path.join(tmp, "mf_A.bin"), 6, 6)
        mfB = load_mat_c64(os.path.join(tmp, "mf_B.bin"), 6, 6)
        assert_close("scipy:schur_complex_eigs", sort_complex(load_vec_c64(os.path.join(tmp, "schur_eigs.bin"))),
                     sort_complex(sla.eigvals(mfA)), 1e-11, results)
        assert_close("scipy:qz_factor_eigs", sort_complex(load_vec_c64(os.path.join(tmp, "qz_eigs.bin"))),
                     sort_complex(sla.eigvals(mfA, mfB)), 1e-10, results)

        eAr = load_mat_f64(os.path.join(tmp, "expm_Ar.bin"), 6, 6)
        assert_close("scipy:expm_f64_high_norm", load_mat_f64(os.path.join(tmp, "expm_Ar_out.bin"), 6, 6),
                     sla.expm(eAr), 1e-11, results)
        assert_close("scipy:expm_c64", load_mat_c64(os.path.join(tmp, "expm_Ac_out.bin"), 6, 6),
                     sla.expm(mfA), 1e-11, results)
        assert_close("scipy:logm_c64", load_mat_c64(os.path.join(tmp, "logm_Ac_out.bin"), 6, 6),
                     sla.logm(mfA), 1e-11, results)
        assert_close("scipy:sqrtm_c64", load_mat_c64(os.path.join(tmp, "sqrtm_Ac_out.bin"), 6, 6),
                     sla.sqrtm(mfA), 1e-11, results)
        assert_close("scipy:funm_sin_c64", load_mat_c64(os.path.join(tmp, "funm_sin_out.bin"), 6, 6),
                     sla.sinm(mfA), 1e-11, results)

        sA4 = load_mat_c64(os.path.join(tmp, "sylv_A.bin"), 4, 4)
        sB3 = load_mat_c64(os.path.join(tmp, "sylv_B.bin"), 3, 3)
        sC43 = load_mat_c64(os.path.join(tmp, "sylv_C.bin"), 4, 3)
        sX43 = load_mat_c64(os.path.join(tmp, "sylv_X.bin"), 4, 3)
        assert_close("scipy:sylvester_c64", sX43, sla.solve_sylvester(sA4, sB3, sC43), 1e-11, results)

        mvA = load_mat_f64(os.path.join(tmp, "expmv_A.bin"), 40, 40)
        mvv = load_vec_f64(os.path.join(tmp, "expmv_v.bin"))
        mvy = load_vec_f64(os.path.join(tmp, "expmv_y.bin"))
        assert_close("scipy:expmv_n40_t15", mvy, sla.expm(15.0 * mvA) @ mvv, 1e-10, results)

        nC = load_mat_f64(os.path.join(tmp, "nnls_C.bin"), 10, 5)
        nd = load_vec_f64(os.path.join(tmp, "nnls_d.bin"))
        nx = load_vec_f64(os.path.join(tmp, "nnls_x.bin"))
        assert_close("scipy:lsqnonneg", nx, nnls(nC, nd)[0], 1e-10, results)

        run_matlab_check(tmp, results)

    print(f"ALL LINALG REFERENCE CHECKS PASSED ({len(results)}/{len(results)}):")
    for name, detail in results:
        print(f"  [PASS] {name} ({detail})")


if __name__ == "__main__":
    main()
