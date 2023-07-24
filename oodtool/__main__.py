import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import sys
from oodtool.pyqt_gui.main_window import OoDToolApp

if __name__ == "__main__":
    app = OoDToolApp(sys.argv)
