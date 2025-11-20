import os

site_packages = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages")
packages = []

for entry in os.listdir(site_packages):
    if entry.endswith(".dist-info"):
        pkg = entry.split("-")[0]
        version = entry.split("-")[1] if "-" in entry else "unknown"
        packages.append(f"{pkg}=={version}")

with open("requirements_extracted.txt", "w") as f:
    f.write("\n".join(packages))

print("Extracted packages written to requirements_extracted.txt")
