
from wzk.logger import log_print
import pikepdf

file = "/Users/maxmuster/Downloads/test.pdf"

for year in range(1950, 2025):
    for month in range(12):
        for day in range(1, 32):
            password = f"{day:02d}.{month:02d}.{year}"
            log_print(password)
            try:
                with pikepdf.open(file, password=password) as pdf:
                    num_pages = len(pdf.pages)
                    input("found")
            except pikepdf._core.PasswordError:
                pass
