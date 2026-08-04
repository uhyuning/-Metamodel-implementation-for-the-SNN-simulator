"""
생성된 C++/SIMD 백엔드 코드를 MSVC로 컴파일하고 실행하는 헬퍼.

vcvars64.bat 환경을 몰라도 이 스크립트 하나로 "JSON 메타모델 -> C++ 코드 생성 ->
컴파일 -> 실행 -> 레이턴시 출력"까지의 마지막 단계를 재현할 수 있게 한다.

사용 예:
    python scripts/build_cpp.py generated_code/CppSimdStandardModel_CppSIMD.cpp
"""
import argparse
import os
import subprocess
import tempfile


def find_vcvars64():
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if not os.path.exists(vswhere):
        return None
    result = subprocess.run(
        [vswhere, "-latest", "-products", "*",
         "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        capture_output=True, text=True,
    )
    install_path = result.stdout.strip()
    if not install_path:
        return None
    vcvars = os.path.join(install_path, "VC", "Auxiliary", "Build", "vcvars64.bat")
    return vcvars if os.path.exists(vcvars) else None


def compile_cpp(cpp_path: str, vec_report: bool = False) -> str:
    """생성된 .cpp를 MSVC로 컴파일하고 실행 파일 경로를 반환한다 (실행은 하지 않음).
    반복 실행으로 지연시간을 여러 번 측정하는 벤치마크에서는 컴파일을 한 번만 하고
    싶으므로 build_and_run과 별도로 노출한다.
    """
    cpp_path = os.path.abspath(cpp_path)
    if not os.path.exists(cpp_path):
        raise FileNotFoundError(cpp_path)

    out_dir = os.path.dirname(cpp_path)
    model_name = os.path.splitext(os.path.basename(cpp_path))[0]
    exe_path = os.path.join(out_dir, model_name + ".exe")

    vcvars = find_vcvars64()
    if vcvars is None:
        raise RuntimeError(
            "MSVC(vcvars64.bat)를 찾을 수 없습니다. "
            "Visual Studio의 'Desktop development with C++' 워크로드가 설치되어 있는지 확인하세요."
        )

    # subprocess에 "cmd /c <중첩따옴표 포함 문자열>"을 리스트로 넘기면 Windows의
    # list2cmdline이 내부 따옴표를 다시 이스케이프해서 깨지므로, 임시 .bat 파일에
    # 명령을 써넣고 그 배치 파일 자체를 인자 없이 실행한다.
    vec_flag = "/Qvec-report:1 " if vec_report else ""
    bat_lines = [
        "@echo off",
        f'call "{vcvars}"',
        # /Fo 값 끝에 백슬래시+따옴표가 오면 Windows 인자 파싱 규칙상 이스케이프된 따옴표로
        # 해석되어 깨지므로 슬래시를 사용한다 (cl.exe는 경로 구분자로 '/'도 허용함).
        f'cl /O2 /fp:fast /arch:AVX2 /EHsc /nologo {vec_flag}"{cpp_path}" '
        f'/Fe:"{exe_path}" /Fo:"{out_dir}/"',
    ]
    fd, bat_path = tempfile.mkstemp(suffix=".bat")
    try:
        with os.fdopen(fd, "w", encoding="cp949") as f:
            f.write("\r\n".join(bat_lines) + "\r\n")
        result = subprocess.run(
            [bat_path], capture_output=True, text=True, encoding="cp949", errors="replace"
        )
    finally:
        os.remove(bat_path)

    if result.returncode != 0 or not os.path.exists(exe_path):
        raise RuntimeError(
            f"컴파일 실패 (exit code {result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
    if vec_report:
        print(result.stdout)

    return exe_path


def run_exe(exe_path: str) -> str:
    run_result = subprocess.run(
        [exe_path], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if run_result.returncode != 0:
        raise RuntimeError(f"실행 실패 (exit code {run_result.returncode}):\n{run_result.stderr}")
    return run_result.stdout


def build_and_run(cpp_path: str, vec_report: bool = False) -> str:
    exe_path = compile_cpp(cpp_path, vec_report=vec_report)
    output = run_exe(exe_path)
    print(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile and run a generated C++/SIMD SNN kernel with MSVC")
    parser.add_argument("cpp_file", help="Path to the generated .cpp file")
    parser.add_argument("--vec-report", action="store_true", help="벡터화 여부를 컴파일러 진단으로 출력")
    args = parser.parse_args()
    build_and_run(args.cpp_file, vec_report=args.vec_report)
