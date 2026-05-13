"""One-off: extract plain text from a .docx to stdout UTF-8."""
import sys
import zipfile
import xml.etree.ElementTree as ET

def main() -> None:
    path = sys.argv[1]
    with zipfile.ZipFile(path, "r") as z:
        xml_content = z.read("word/document.xml")
    root = ET.fromstring(xml_content)
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    parts: list[str] = []
    for t in root.iter(f"{{{ns}}}t"):
        if t.text:
            parts.append(t.text)
        if t.tail:
            parts.append(t.tail)
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write("".join(parts))


if __name__ == "__main__":
    main()
