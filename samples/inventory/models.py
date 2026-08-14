from pricing import apply_discount


class Item:
    def __init__(self, name, price, quantity, discount_percent=0):
        self.name = name
        self.price = price
        self.quantity = quantity
        self.discount_percent = discount_percent

    def total_value(self):
        """Total value of this item after discount, times quantity."""
        unit_price = apply_discount(self.price, self.discount_percent)
        return unit_price * self.quantity
