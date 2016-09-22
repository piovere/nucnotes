from tabulate import tabulate

s_fuel = 7.68
s_moderator = {
                "Graphite": 0.004,
                "Beryllium": 0.010,
                "Water": 0.66,
                "Heavywater": 0.001
              }
eta = 2.068


def moderator_ratio(sigma_moderator):
    return (eta - 1.0) * (s_fuel / sigma_moderator)

headers = ["Moderator", "Moderator/Fuel Number Ratio"]
results = []

for moderator, sigma in s_moderator.items():
    results.append([moderator, moderator_ratio(sigma)])

print tabulate(results, headers=headers)
