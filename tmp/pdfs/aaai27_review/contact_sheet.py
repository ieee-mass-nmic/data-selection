from pathlib import Path
from PIL import Image, ImageDraw


def make_sheet(source: Path, output: Path, columns: int = 3) -> None:
    paths = sorted(source.glob("page-*.png"))
    images = [Image.open(path).convert("RGB") for path in paths]
    thumb_width = 420
    thumb_height = round(images[0].height * thumb_width / images[0].width)
    label_height = 28
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (path, page) in enumerate(zip(paths, images)):
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        sheet.paste(page.resize((thumb_width, thumb_height)), (x, y + label_height))
        draw.text((x + 8, y + 6), path.stem, fill="black")
    sheet.save(output, quality=95)


root = Path(__file__).parent
make_sheet(root / "main", root / "main-contact.jpg")
make_sheet(root / "supp", root / "supp-contact.jpg")
