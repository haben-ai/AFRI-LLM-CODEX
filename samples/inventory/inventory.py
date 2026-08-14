from models import Item


class Inventory:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)

    def compute_total(self):
        """Sum the total value of every item -- buggy: averages instead of
        summing, and crashes with ZeroDivisionError on an empty inventory."""
        total = 0
        for item in self.items:
            total += item.total_value()
        return total / len(self.items)
