from pathlib import Path
import os

def find_data_path(data_dir):
    """
            Automatically detect relevant file path for data loading based on OS.
            For Windows, checks common drive letters and user paths.
            For Linux, uses the data_dir as is or checks common mount points.
            """
    # Detect operating system
    if os.name == 'nt':  # Windows
        target_subdir = data_dir
        # Windows candidate paths
        candidate_roots = [
            Path("F:/"),
            Path("C:/Users/Grant"),
            Path("C:/Users/gkirc")
        ]

        for root in candidate_roots:
            candidate = root / target_subdir
            if candidate.exists():
                print(f"Detected Windows data directory: {candidate}")
                return str(candidate)

        raise FileNotFoundError("Could not locate the Windows data directory.")

    else:  # Linux/Unix
        # For Linux, first try the direct path
        linux_path = Path(data_dir)
        if linux_path.exists():
            print(f"Using Linux data directory: {linux_path}")
            return str(linux_path)

        # If direct path doesn't exist, check common Linux mount points
        linux_candidates = [
            Path("/home/grki4829/Data"),
            Path("/data"),
            Path("/mnt/data"),
        ]

        for candidate in linux_candidates:
            if candidate.exists():
                print(f"Detected Linux data directory: {candidate}")
                return str(candidate)

        raise FileNotFoundError("Could not locate the Linux data directory.")
