# ═══════════════════════════════════════════════════════════════
#  FaceSwap AI — Test Report Generator
#  Generates HTML + JSON reports with pass/fail per module
# ═══════════════════════════════════════════════════════════════
import json
import os
import sys
import time
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import defaultdict


# ── Module definitions ────────────────────────────────────────
MODULES = [
    ("01", "Webcam Capture", "test_01_webcam_capture.py"),
    ("02", "WebSocket Stability", "test_02_websocket_stability.py"),
    ("03", "AI Inference", "test_03_ai_inference.py"),
    ("04", "Face Detection Accuracy", "test_04_face_detection.py"),
    ("05", "Face Tracking", "test_05_face_tracking.py"),
    ("06", "Multiple Face Support", "test_06_multiple_faces.py"),
    ("07", "Low Light", "test_07_low_light.py"),
    ("08", "Side Face", "test_08_side_face.py"),
    ("09", "Glasses", "test_09_glasses.py"),
    ("10", "Beard", "test_10_beard.py"),
    ("11", "Different Resolutions", "test_11_resolutions.py"),
    ("12", "GPU Overload", "test_12_gpu_overload.py"),
    ("13", "Network Interruption", "test_13_network_interruption.py"),
    ("14", "Memory Leak", "test_14_memory_leak.py"),
    ("15", "Frame Drops", "test_15_frame_drops.py"),
    ("16", "Long Session Stability", "test_16_long_session.py"),
]

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(BACKEND_DIR, "tests")
REPORTS_DIR = os.path.join(BACKEND_DIR, "logs", "test-reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def run_module(module_num, module_name, module_file):
    """Run a single test module and capture results."""
    print(f"\n{'='*60}")
    print(f"  Module {module_num}: {module_name}")
    print(f"{'='*60}")

    xml_path = os.path.join(REPORTS_DIR, f"junit-{module_num}.xml")
    module_path = os.path.join(TESTS_DIR, module_file)

    cmd = [
        sys.executable, "-m", "pytest",
        module_path,
        "-v",
        "--tb=short",
        f"--junitxml={xml_path}",
        "--no-header",
        "-q",
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BACKEND_DIR)
    elapsed = time.time() - start

    # Parse JUnit XML
    tests_run = 0
    failures = 0
    errors = 0
    skipped = 0
    test_details = []

    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            tests_run = int(root.get("tests", 0))
            failures = int(root.get("failures", 0))
            errors = int(root.get("errors", 0))
            skipped = int(root.get("skipped", 0))

            for testcase in root.iter("testcase"):
                name = testcase.get("name", "")
                classname = testcase.get("classname", "")
                time_s = float(testcase.get("time", 0))

                status = "passed"
                message = ""
                for child in testcase:
                    if child.tag == "failure":
                        status = "failed"
                        message = child.get("message", "")
                    elif child.tag == "error":
                        status = "error"
                        message = child.get("message", "")
                    elif child.tag == "skipped":
                        status = "skipped"
                        message = child.get("message", "")

                test_details.append({
                    "name": name,
                    "classname": classname,
                    "status": status,
                    "time_s": round(time_s, 3),
                    "message": message[:200],
                })
        except Exception as e:
            print(f"  ⚠ XML parse error: {e}")

    passed = tests_run - failures - errors - skipped
    status = "PASS" if failures == 0 and errors == 0 else "FAIL"

    print(f"  Result: {status} | {passed} passed, {failures} failed, "
          f"{errors} errors, {skipped} skipped | {elapsed:.1f}s")

    return {
        "module_num": module_num,
        "module_name": module_name,
        "module_file": module_file,
        "status": status,
        "tests_run": tests_run,
        "passed": passed,
        "failed": failures,
        "errors": errors,
        "skipped": skipped,
        "duration_s": round(elapsed, 2),
        "stdout": result.stdout[-2000:] if result.stdout else "",
        "stderr": result.stderr[-1000:] if result.stderr else "",
        "test_details": test_details,
    }


def generate_json_report(results):
    """Generate JSON report."""
    total_tests = sum(r["tests_run"] for r in results)
    total_passed = sum(r["passed"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    total_duration = sum(r["duration_s"] for r in results)
    modules_passed = sum(1 for r in results if r["status"] == "PASS")
    modules_failed = sum(1 for r in results if r["status"] == "FAIL")

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_modules": len(results),
            "modules_passed": modules_passed,
            "modules_failed": modules_failed,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_errors": total_errors,
            "total_skipped": total_skipped,
            "total_duration_s": round(total_duration, 2),
            "overall_status": "PASS" if modules_failed == 0 else "FAIL",
        },
        "modules": results,
    }

    json_path = os.path.join(REPORTS_DIR, "test-report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  JSON report: {json_path}")
    return report


def generate_html_report(results, report_data):
    """Generate HTML report with pass/fail per module."""
    summary = report_data["summary"]

    # Module rows
    rows_html = ""
    for r in results:
        status_color = "#22c55e" if r["status"] == "PASS" else "#ef4444"
        status_bg = "rgba(34,197,94,0.1)" if r["status"] == "PASS" else "rgba(239,68,68,0.1)"

        # Test detail rows
        details_html = ""
        for td in r["test_details"]:
            td_color = {"passed": "#22c55e", "failed": "#ef4444",
                        "error": "#f59e0b", "skipped": "#6b7280"}[td["status"]]
            details_html += f"""
            <tr>
                <td style="padding:4px 8px;font-family:monospace;font-size:12px;">{td["name"]}</td>
                <td style="padding:4px 8px;"><span style="color:{td_color};font-weight:600;">{td["status"].upper()}</span></td>
                <td style="padding:4px 8px;text-align:right;font-family:monospace;font-size:12px;">{td["time_s"]}s</td>
                <td style="padding:4px 8px;font-size:11px;color:#9ca3af;">{td["message"]}</td>
            </tr>"""

        expandable = f"""
        <details style="margin-top:8px;">
            <summary style="cursor:pointer;color:#6b7280;font-size:12px;">
                Show {len(r["test_details"])} test cases
            </summary>
            <table style="width:100%;margin-top:8px;border-collapse:collapse;">
                <thead>
                    <tr style="border-bottom:1px solid #374151;text-align:left;">
                        <th style="padding:4px 8px;font-size:11px;color:#9ca3af;">TEST</th>
                        <th style="padding:4px 8px;font-size:11px;color:#9ca3af;">STATUS</th>
                        <th style="padding:4px 8px;font-size:11px;color:#9ca3af;text-align:right;">TIME</th>
                        <th style="padding:4px 8px;font-size:11px;color:#9ca3af;">MESSAGE</th>
                    </tr>
                </thead>
                <tbody>{details_html}</tbody>
            </table>
        </details>""" if details_html else ""

        rows_html += f"""
        <div style="background:{status_bg};border:1px solid {status_color}33;border-radius:12px;padding:20px;margin-bottom:12px;">
            <div style="display:flex;align-items:center;justify-content:space-between;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="width:32px;height:32px;border-radius:50%;background:{status_color};display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:14px;">
                        {r["module_num"]}
                    </div>
                    <div>
                        <div style="font-weight:600;font-size:16px;color:#e5e7eb;">{r["module_name"]}</div>
                        <div style="font-size:12px;color:#6b7280;font-family:monospace;">{r["module_file"]}</div>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:24px;">
                    <div style="text-align:center;">
                        <div style="font-size:20px;font-weight:700;color:#22c55e;">{r["passed"]}</div>
                        <div style="font-size:10px;color:#6b7280;text-transform:uppercase;">Passed</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:20px;font-weight:700;color:#ef4444;">{r["failed"]}</div>
                        <div style="font-size:10px;color:#6b7280;text-transform:uppercase;">Failed</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:20px;font-weight:700;color:#f59e0b;">{r["errors"]}</div>
                        <div style="font-size:10px;color:#6b7280;text-transform:uppercase;">Errors</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:20px;font-weight:700;color:#6b7280;">{r["skipped"]}</div>
                        <div style="font-size:10px;color:#6b7280;text-transform:uppercase;">Skipped</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:20px;font-weight:700;color:#9ca3af;">{r["duration_s"]}s</div>
                        <div style="font-size:10px;color:#6b7280;text-transform:uppercase;">Duration</div>
                    </div>
                    <div style="padding:6px 16px;border-radius:8px;background:{status_color};color:white;font-weight:700;font-size:14px;">
                        {r["status"]}
                    </div>
                </div>
            </div>
            {expandable}
        </div>"""

    overall_color = "#22c55e" if summary["overall_status"] == "PASS" else "#ef4444"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FaceSwap AI — Test Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1117;
            color: #e5e7eb;
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 4px;
            background: linear-gradient(135deg, #2dd4bf, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .timestamp {{ color: #6b7280; font-size: 13px; margin-bottom: 32px; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .summary-card {{
            background: #1a1d28;
            border: 1px solid #2a2d3a;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .summary-card .value {{ font-size: 32px; font-weight: 700; }}
        .summary-card .label {{ font-size: 11px; color: #6b7280; text-transform: uppercase; margin-top: 4px; }}
        .overall-badge {{
            display: inline-block;
            padding: 8px 24px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 16px;
            margin-bottom: 24px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 FaceSwap AI — Test Report</h1>
        <div class="timestamp">Generated: {report_data["generated_at"]} | Duration: {summary["total_duration_s"]}s</div>

        <div class="overall-badge" style="background:{overall_color};color:white;">
            {summary["overall_status"]} — {summary["modules_passed"]}/{summary["total_modules"]} modules passed
        </div>

        <div class="summary-grid">
            <div class="summary-card">
                <div class="value" style="color:#e5e7eb;">{summary["total_modules"]}</div>
                <div class="label">Modules</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color:#e5e7eb;">{summary["total_tests"]}</div>
                <div class="label">Total Tests</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color:#22c55e;">{summary["total_passed"]}</div>
                <div class="label">Passed</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color:#ef4444;">{summary["total_failed"]}</div>
                <div class="label">Failed</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color:#f59e0b;">{summary["total_errors"]}</div>
                <div class="label">Errors</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color:#6b7280;">{summary["total_skipped"]}</div>
                <div class="label">Skipped</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color:#9ca3af;">{summary["total_duration_s"]}s</div>
                <div class="label">Duration</div>
            </div>
        </div>

        {rows_html}
    </div>
</body>
</html>"""

    html_path = os.path.join(REPORTS_DIR, "test-report.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  HTML report: {html_path}")


def generate_console_report(results, report_data):
    """Print a console summary table."""
    summary = report_data["summary"]

    print("\n" + "=" * 80)
    print("  🎭 FaceSwap AI — TEST REPORT SUMMARY")
    print("=" * 80)
    print(f"  Generated: {report_data['generated_at']}")
    print()

    # Module table
    print(f"  {'#':<4} {'Module':<30} {'Status':<8} {'Pass':>6} {'Fail':>6} {'Err':>6} {'Skip':>6} {'Time':>8}")
    print("  " + "-" * 76)

    for r in results:
        status_str = f"\033[92m{r['status']}\033[0m" if r["status"] == "PASS" else f"\033[91m{r['status']}\033[0m"
        print(f"  {r['module_num']:<4} {r['module_name']:<30} {status_str:<8} "
              f"{r['passed']:>6} {r['failed']:>6} {r['errors']:>6} {r['skipped']:>6} "
              f"{r['duration_s']:>7.1f}s")

    print("  " + "-" * 76)
    overall = summary["overall_status"]
    overall_str = f"\033[92m{overall}\033[0m" if overall == "PASS" else f"\033[91m{overall}\033[0m"
    print(f"  {'':4} {'TOTAL':<30} {overall_str:<8} "
          f"{summary['total_passed']:>6} {summary['total_failed']:>6} "
          f"{summary['total_errors']:>6} {summary['total_skipped']:>6} "
          f"{summary['total_duration_s']:>7.1f}s")
    print()
    print(f"  Modules: {summary['modules_passed']}/{summary['total_modules']} passed | "
          f"Tests: {summary['total_passed']}/{summary['total_tests']} passed")
    print("=" * 80)


def main():
    """Run all test modules and generate reports."""
    print("\n" + "=" * 80)
    print("  🎭 FaceSwap AI — Comprehensive Test Suite")
    print("  16 modules | " + str(sum(1 for _ in MODULES)) + " test categories")
    print("=" * 80)

    results = []
    for module_num, module_name, module_file in MODULES:
        result = run_module(module_num, module_name, module_file)
        results.append(result)

    # Generate reports
    print("\n" + "=" * 80)
    print("  Generating reports...")
    report_data = generate_json_report(results)
    generate_html_report(results, report_data)
    generate_console_report(results, report_data)

    # Exit code
    exit(0 if report_data["summary"]["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()