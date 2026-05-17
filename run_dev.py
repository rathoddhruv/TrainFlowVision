import subprocess
import sys
import os
import socket
import psutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BE_DIR = ROOT_DIR / "BE"
FE_DIR = ROOT_DIR / "FE"
ML_DIR = ROOT_DIR / "ml"

# Ensure Python can find local modules
os.environ["PYTHONPATH"] = str(ROOT_DIR)

def check_port(port):
    """Check if a port is in use to prevent silent startup failures."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def resolve_port_conflict(port):
    """Detect and optionally resolve port conflicts automatically."""
    if not check_port(port):
        return True

    print(f"\n[WARNING] Port {port} is currently in use.")
    
    occupying_proc = None
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    occupying_proc = proc
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        if occupying_proc:
            break
            
    if occupying_proc:
        print(f"Port {port} is currently used by:")
        print(f"PID: {occupying_proc.pid}")
        print(f"Process: {occupying_proc.info['name']}")
        
        choice = input("Kill this process automatically? [Y/N]: ").strip().lower()
        if choice in ('y', 'yes'):
            try:
                # Kill process tree safely (important for uvicorn reloader)
                procs_to_kill = occupying_proc.children(recursive=True)
                procs_to_kill.append(occupying_proc)
                for p in procs_to_kill:
                    p.terminate()
                
                gone, alive = psutil.wait_procs(procs_to_kill, timeout=3)
                for p in alive:
                    p.kill()
                    
                print(f"Successfully killed process {occupying_proc.pid} and its children.")
                import time
                time.sleep(1)
                if check_port(port):
                    print(f"[ERROR] Port {port} is still occupied after killing process.")
                    sys.exit(1)
                return True
            except Exception as e:
                print(f"[ERROR] Failed to kill process: {e}")
                sys.exit(1)
        else:
            print("Startup aborted by user.")
            sys.exit(0)
    else:
        print(f"Port {port} appears occupied but no visible process owns it.")
        print("This may be a Windows stale socket issue (HNS/Hyper-V).")
        print("Try running as Administrator:")
        print("  net stop hns")
        print("  net start hns")
        print("or restart Windows.")
        sys.exit(1)

def show_cuda_status():
    """Verify hardware acceleration availability for YOLO training."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] CUDA not available. Running on CPU.")
    except ImportError:
        print("[WARN] PyTorch not installed yet. Skipping CUDA check.")

def validate_structure():
    """Ensure the repo hasn't been improperly cloned or missing vital folders."""
    required_dirs = [BE_DIR, FE_DIR, ML_DIR]
    for d in required_dirs:
        if not d.exists():
            print(f"[ERROR] Required directory {d.name} is missing!")
            sys.exit(1)

    fe_pkg = FE_DIR / "package.json"
    if not fe_pkg.exists():
        print("[ERROR] FE/package.json not found! Ensure Angular frontend is present.")
        sys.exit(1)

def install_dependencies():
    """Install backend dependencies if not already present."""
    print("Checking backend dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True
    )
    
    """Install frontend dependencies only if node_modules is missing."""
    print("Checking frontend dependencies...")
    node_modules = FE_DIR / "node_modules"
    if not node_modules.exists():
        print("node_modules missing. Installing FE dependencies...")
        shell = True if os.name == 'nt' else False
        pkg_lock = FE_DIR / "package-lock.json"
        cmd = ["npm", "ci"] if pkg_lock.exists() else ["npm", "install"]
        subprocess.run(cmd, cwd=FE_DIR, shell=shell, check=True)

def run_dev():
    print("Starting TrainFlowVision Fullstack...")
    
    validate_structure()
    install_dependencies()
    show_cuda_status()

    # Port checks and conflict resolution
    resolve_port_conflict(8000)
    resolve_port_conflict(4200)

    # Start Backend
    print("Starting FastAPI Backend (Port 8000)...")
    be_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "BE.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        cwd=ROOT_DIR
    )

    # Start Frontend
    print("Starting Angular Frontend (Port 4200)...")
    shell = True if os.name == 'nt' else False
    
    # Disable Angular Analytics prompt to prevent hanging
    fe_env = os.environ.copy()
    fe_env["NG_CLI_ANALYTICS"] = "false"
    
    fe_process = subprocess.Popen(
        ["npm", "start"],
        cwd=FE_DIR,
        shell=shell,
        env=fe_env
    )

    print("Waiting for services to initialize...")

    try:
        be_process.wait()
        fe_process.wait()
    except KeyboardInterrupt:
        print("\nStopping services cleanly...")
        be_process.terminate()
        fe_process.terminate()
        be_process.wait()
        fe_process.wait()
        print("Goodbye!")

if __name__ == "__main__":
    run_dev()
