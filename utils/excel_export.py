import pandas as pd

def export_to_excel(cost_data, path):
    rows = []

    for item, data in cost_data['details'].items():
        rows.append([
            item,
            data['count'],
            data['unit_price'],
            data['total']
        ])

    df = pd.DataFrame(rows, columns=[
        'Component', 'Count', 'Unit Price', 'Total Cost'
    ])

    df.to_excel(path, index=False)