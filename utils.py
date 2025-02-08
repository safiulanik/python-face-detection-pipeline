import sys
import subprocess


def open_image(path):
    image_viewer_from_command_line = {
        'linux': 'xdg-open',
        'win32': 'explorer',
        'darwin': 'open'
    }[sys.platform]
    subprocess.run([image_viewer_from_command_line, path])
