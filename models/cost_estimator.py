PRICES = {
    'Pipe': 100,
    'Valve': 250,
    'Duct': 500,
    'Diffuser': 300
}

def calculate_cost(counts):
    total_cost = 0
    detailed = {}

    for item, count in counts.items():
        cost = PRICES.get(item, 0) * count
        detailed[item] = {
            'count': count,
            'unit_price': PRICES.get(item, 0),
            'total': cost
        }
        total_cost += cost

    return {
        'details': detailed,
        'total_cost': total_cost
    }