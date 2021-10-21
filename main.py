import os
import sys
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, 'work_object_detection'))

from work_object_detection import work_object_detection_server

# Run debug mode
# python3 -d main.py


def main():
    work_object_detection_server.run_server()
    return


if __name__ == "__main__":
    main()
