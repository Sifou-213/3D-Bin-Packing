"""Excel load-plan report in the same dark theme as the web app.

The payload comes from the browser (static/js/app.js, exportExcel) and already contains
everything the report shows: vehicle, totals, one picture per truck, the items and every
box in loading order.
"""
import base64
import io
from datetime import datetime

from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# Theme (same tokens as static/css/app.css)
BG = '0D0F12'
SURFACE = '12151A'
LINE = '23272E'
BRIGHT = 'F2F3F5'
TEXT = 'E8EAED'
MUTED = '9AA1AB'
FAINT = '6E7680'
MONO = 'Consolas'      # closest mono font installed with Excel to the site's JetBrains Mono
SANS = 'Segoe UI'

MODE_LABEL = {'truck': 'Un camion', 'fleet': 'Plusieurs camions', 'pallet': 'Une palette'}
QUALITY_LABEL = {'fast': 'Rapide', 'normal': 'Normale', 'best': 'Soignée'}
IMAGE_WIDTH_PX = 760
ROW_PX = 20  # default row height (15 pt) in pixels


def _fill(color):
    return PatternFill('solid', fgColor=color)


def _label(bold=False, color=MUTED, size=10):
    return Font(name=MONO, size=size, bold=bold, color=color)


def _text(bold=False, color=TEXT, size=11):
    return Font(name=SANS, size=size, bold=bold, color=color)


DASH = Side(style='dashed', color=LINE)
THIN = Side(style='thin', color=LINE)


def _fr(value, digits=0):
    """French number formatting for text cells (1 234,5)."""
    s = f'{value:,.{digits}f}'
    return s.replace(',', ' ').replace('.', ',')


def _paint(ws, rows, cols):
    """Gives the whole visible area the dark background and hides the grid."""
    ws.sheet_view.showGridLines = False
    for row in ws.iter_rows(min_row=1, max_row=rows, min_col=1, max_col=cols):
        for cell in row:
            cell.fill = _fill(BG)
            cell.font = _text()


def _dashed_rule(ws, row, first_col, last_col):
    for col in range(first_col, last_col + 1):
        ws.cell(row=row, column=col).border = Border(bottom=DASH)


def _header_row(ws, row, headers, first_col=2):
    for i, title in enumerate(headers):
        cell = ws.cell(row=row, column=first_col + i, value=title.upper())
        cell.font = _label()
        cell.fill = _fill(BG)
        cell.border = Border(bottom=DASH)
        cell.alignment = Alignment(vertical='center')
    ws.row_dimensions[row].height = 22


def _swatch(cell, color):
    cell.fill = _fill(color.lstrip('#').upper())
    cell.border = Border(left=Side(style='thick', color=BG), right=Side(style='thick', color=BG),
                         top=Side(style='thick', color=BG), bottom=Side(style='thick', color=BG))


def _title_block(ws, payload, last_col):
    ws.row_dimensions[1].height = 10
    left = ws.cell(row=2, column=2, value='PLAN DE CHARGEMENT')
    left.font = _label(size=11)
    right = ws.cell(row=2, column=last_col, value=payload.get('vehicle', '').upper())
    right.font = _label(bold=True, color=BRIGHT, size=11)
    right.alignment = Alignment(horizontal='right')
    ws.row_dimensions[2].height = 26
    _dashed_rule(ws, 2, 2, last_col)


def _plan_sheet(ws, payload):
    last_col = 9
    for col in range(2, last_col + 1):
        ws.column_dimensions[get_column_letter(col)].width = 13
    ws.column_dimensions['A'].width = 3
    ws.column_dimensions['B'].width = 4      # colour swatch
    ws.column_dimensions['C'].width = 46     # names

    views = payload.get('views', [])
    items = payload.get('items', [])
    rows_needed = 12 + len(views) * 26 + len(items) + 10
    _paint(ws, rows_needed, last_col + 2)
    _title_block(ws, payload, last_col)

    # Stats line, like the panel footer: COLIS 56/56   VOLUME CHARGÉ 48 %
    stats = [('COLIS', f"{payload.get('packed', 0)}/{payload.get('total', 0)}"),
             ('VOLUME CHARGÉ', f"{_fr(payload.get('fill', 0))} %")]
    if len(views) > 1:
        stats.append(('CAMIONS', str(len(views))))
    line = '   '.join(f'{name} {value}' for name, value in stats)
    ws.cell(row=4, column=2, value=line).font = _label(bold=True, color=BRIGHT)
    ws.row_dimensions[4].height = 24

    left = payload.get('total', 0) - payload.get('packed', 0)
    if left <= 0:
        verdict = f"Tout rentre ! {payload.get('total', 0)} colis rangés."
    else:
        verdict = f"{left} colis ne rentre{'nt' if left > 1 else ''} pas."
    cell = ws.cell(row=6, column=2, value=verdict)
    cell.font = _text(bold=True, color=BRIGHT, size=14)
    ws.row_dimensions[6].height = 26
    note = ws.cell(row=7, column=2, value='Le chargement se construit couche par couche, du fond du camion vers la porte. '
                                          "L'ordre exact est dans l'onglet « Ordre de chargement ».")
    note.font = _text(color=MUTED, size=10)

    row = 9
    for view in views:
        title = ws.cell(row=row, column=2, value=f"{view.get('title', '').upper()} · {_fr(view.get('fill', 0))} % · {len(view.get('boxes', []))} COLIS")
        title.font = _label(bold=True, color=BRIGHT)
        _dashed_rule(ws, row, 2, last_col)
        row += 1
        rows_used = 2
        image = _image(view.get('image'))
        if image is not None:
            ws.add_image(image, f'B{row}')
            rows_used = int(image.height / ROW_PX) + 2
        row += rows_used

    # Items table
    row += 1
    ws.cell(row=row, column=2, value='LES COLIS').font = _label(bold=True, color=BRIGHT)
    row += 1
    _header_row(ws, row, ['', 'Colis', 'Taille (cm)', 'Poids (kg)', 'Demandés', 'Rangés', 'Hors', 'Options'])
    for c in range(5, 9):
        ws.cell(row=row, column=c).alignment = Alignment(horizontal='center', vertical='center')
    for it in items:
        row += 1
        _swatch(ws.cell(row=row, column=2), it.get('color', '#9A9A9A'))
        ws.cell(row=row, column=3, value=it.get('name')).font = _text(color=BRIGHT)
        size = ' × '.join(_fr((it.get(k) or 0) * 100) for k in ('d', 'w', 'h'))
        ws.cell(row=row, column=4, value=size).font = _label(color=TEXT)
        _num(ws.cell(row=row, column=5, value=it.get('weight', 0)), '0.0')
        _num(ws.cell(row=row, column=6, value=it.get('qty', 0)), '0')
        _num(ws.cell(row=row, column=7, value=it.get('packed', 0)), '0', bold=True)
        _num(ws.cell(row=row, column=8, value=it.get('left', 0)), '0', color=MUTED if not it.get('left') else BRIGHT)
        for c in range(5, 9):
            ws.cell(row=row, column=c).alignment = Alignment(horizontal='center')
        options = ', '.join(o for o, on in (('toujours debout', it.get('upright')), ('fragile', it.get('fragile'))) if on) or '–'
        ws.cell(row=row, column=9, value=options).font = _text(color=MUTED, size=10)
        for c in range(3, last_col + 1):
            ws.cell(row=row, column=c).border = Border(bottom=THIN)
        ws.row_dimensions[row].height = 20


def _num(cell, fmt, bold=False, color=None):
    cell.number_format = fmt
    cell.font = _label(bold=bold, color=color or (BRIGHT if bold else TEXT))
    cell.alignment = Alignment(horizontal='right')


def _image(data_url):
    if not data_url or ',' not in data_url:
        return None
    try:
        img = Image(io.BytesIO(base64.b64decode(data_url.split(',', 1)[1])))
    except Exception:
        return None
    ratio = IMAGE_WIDTH_PX / img.width
    img.width = IMAGE_WIDTH_PX
    img.height = int(img.height * ratio)
    return img


def _order_sheet(ws, payload):
    headers = ['N°', '', 'Colis', 'Camion', 'Longueur (cm)', 'Largeur (cm)', 'Hauteur (cm)',
               'Depuis le fond (m)', 'Depuis le côté (m)', 'Hauteur posée (m)', 'Poids (kg)']
    widths = [3, 7, 4, 44, 12, 14, 14, 14, 18, 18, 18, 12]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    boxes = [(view.get('title', ''), b) for view in payload.get('views', []) for b in view.get('boxes', [])]
    last_col = len(headers) + 1
    _paint(ws, len(boxes) + 12, last_col + 2)
    _title_block(ws, payload, last_col)
    note = ws.cell(row=4, column=2, value='Posez les colis dans cet ordre : chacun repose sur des colis déjà posés.')
    note.font = _text(color=MUTED, size=10)
    _header_row(ws, 6, headers)
    ws.freeze_panes = 'A7'
    for i, (truck, b) in enumerate(boxes):
        r = 7 + i
        shade = _fill(SURFACE if i % 2 else BG)
        for c in range(2, last_col + 1):
            ws.cell(row=r, column=c).fill = shade
        _num(ws.cell(row=r, column=2, value=b.get('step')), '0', bold=True)
        ws.cell(row=r, column=2).alignment = Alignment(horizontal='left')
        _swatch(ws.cell(row=r, column=3), b.get('color', '#9A9A9A'))
        ws.cell(row=r, column=4, value=b.get('name')).font = _text(color=BRIGHT)
        ws.cell(row=r, column=5, value=truck.upper()).font = _label()
        for c, key in ((6, 'dx'), (7, 'dy'), (8, 'dz')):
            _num(ws.cell(row=r, column=c, value=round((b.get(key) or 0) * 100, 1)), '0')
        for c, key in ((9, 'x'), (10, 'y'), (11, 'z')):
            _num(ws.cell(row=r, column=c, value=round(b.get(key) or 0, 3)), '0.00')
        _num(ws.cell(row=r, column=12, value=b.get('weight', 0)), '0.0')
        ws.row_dimensions[r].height = 19


def _settings_sheet(ws, payload, created):
    ws.column_dimensions['A'].width = 3
    ws.column_dimensions['B'].width = 30
    ws.column_dimensions['C'].width = 50
    _paint(ws, 20, 6)
    _title_block(ws, payload, 3)
    rows = [
        ('Chargement', MODE_LABEL.get(payload.get('mode'), '–')),
        ('Véhicule', payload.get('vehicle', '–')),
        ('Recherche', QUALITY_LABEL.get(payload.get('quality'), '–')),
        ('Colis rangés', f"{payload.get('packed', 0)} sur {payload.get('total', 0)}"),
        ('Volume chargé', f"{_fr(payload.get('fill', 0), 1)} %"),
        ('Rapport créé le', created.strftime('%d/%m/%Y à %H:%M')),
    ]
    for i, (k, v) in enumerate(rows):
        r = 4 + i
        ws.cell(row=r, column=2, value=k.upper()).font = _label()
        ws.cell(row=r, column=3, value=v).font = _label(bold=True, color=BRIGHT)
        ws.cell(row=r, column=2).border = ws.cell(row=r, column=3).border = Border(bottom=DASH)
        ws.row_dimensions[r].height = 24


def build_report(payload, created=None):
    """Returns an openpyxl Workbook for the given report payload."""
    created = created or datetime.now()
    wb = Workbook()
    plan = wb.active
    plan.title = 'Plan'
    _plan_sheet(plan, payload)
    _order_sheet(wb.create_sheet('Ordre de chargement'), payload)
    _settings_sheet(wb.create_sheet('Paramètres'), payload, created)
    for ws in wb.worksheets:
        ws.sheet_properties.tabColor = BG
        # Print on one page width, as many pages tall as needed
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.print_options.horizontalCentered = True
    return wb
